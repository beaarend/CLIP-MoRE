import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Tuple

from morelib.triton import monarch_kernel
from morelib.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply, single_monarch_mult
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
This code is adapted from CLIP-LoRA (https://github.com/MaxZanella/CLIP-LoRA) by Max Zanella and 
from MoRE-Framework (https://github.com/SprocketLab/sparse_matrix_fine_tuning) by SprocketLab.
"""

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class BaseLayer():
    def __init__(self, num_blocks, block_rank, block_size = None, dropout_rate: float = 0.0):
        self.num_blocks = num_blocks
        self.block_rank = block_rank
        self.dropout_rate = dropout_rate
        self.block_size = block_size if block_size is not None else 512  # Default block size

    def register_monarch_param(self, dtype: torch.dtype):
        """Register MoRe block-diagonal matrices for each adapted weight."""

        for param_name, monarch_name in self.params_with_monarch.items():
            base_param = getattr(self, param_name)
            
            self.register_parameter(
                f'{monarch_name}_blkdiag1',
                nn.Parameter(torch.empty(
                    self.num_blocks, self.block_rank, self.block_size,
                    device=device, dtype=dtype
                ))
            )
            self.register_parameter(
                f'{monarch_name}_blkdiag2',
                nn.Parameter(torch.empty(
                    self.num_blocks, self.block_size, self.block_rank,
                    device=device, dtype=dtype
                ))
            )
            
            # freezes original weight
            base_param.requires_grad = False

    def init_monarch_param(self, init_type):
        """Initialize the MoRe block-diagonal matrices."""

        for param_name, monarch_name in self.params_with_monarch.items():
            blkdiag1 = getattr(self, f'{monarch_name}_blkdiag1')
            blkdiag2 = getattr(self, f'{monarch_name}_blkdiag2')
            
            if init_type == 'kaiming':
                for factor_name in [f'{monarch_name}_blkdiag1', f'{monarch_name}_blkdiag2']:
                    blkdiag = getattr(self, factor_name)
                    fan_in = blkdiag.shape[-1]
                    gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
                    std = gain / math.sqrt(fan_in)
                    bound = math.sqrt(3.0) * std
                    with torch.no_grad():
                        blkdiag.uniform_(-bound, bound)
            elif init_type == 'zero':
                nn.init.zeros_(blkdiag1)
                nn.init.zeros_(blkdiag2)
            else:
                raise ValueError(f"Unsupported initialization type: {init_type}")

class MonarchLayer(nn.Linear, BaseLayer):
    def __init__(self, existing_linear, num_blocks: int, block_rank: int, dropout_rate: float = 0.0, scaling: float = 1.0):
        self.in_features = existing_linear.in_features
        self.out_features = existing_linear.out_features
        self.num_blocks = num_blocks
        self.block_rank = block_rank
        # self.monarch_impl = monarch_kernel
        self.monarch_impl = blockdiag_butterfly_multiply
        self.block_size = int(math.ceil(self.in_features / self.num_blocks))
        self.merged = False
        self.device = device
        self.dtype = existing_linear.weight.dtype
        self.scaling = scaling

        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features
        )

        # Throw away blocks that are fully padded
        if ((self.num_blocks * self.block_size) > self.in_features):
            self.num_blocks = (self.in_features + self.block_size - 1) // self.block_size
        elif ((self.num_blocks * self.block_size) < self.in_features):
            self.num_blocks = (self.in_features + self.block_size - 1) // self.block_size

        self.load_state_dict(existing_linear.state_dict())
        BaseLayer.__init__(self, num_blocks, block_rank, self.block_size, dropout_rate)

        self.params_with_monarch = {'weight': 'w'}
        self.register_monarch_param(self.dtype)
        self.init_monarch_param('kaiming')

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

    def monarch_forward(self, x: torch.Tensor, blkdiag1: torch.Tensor, blkdiag2: torch.Tensor, use_triton: bool = False) -> torch.Tensor:
        """
        Forward pass using two monarch factors.
        """
        monarch_impl = monarch_kernel
        output = self.monarch_impl(self.preprocess(x), blkdiag1, blkdiag2)
        return self.dropout(self.postprocess(output))
    
    def train(self, mode: bool = True):
        """
        Set the training mode for the Monarch layer.
        """
        super().train(mode)
        if mode:
            # in training mode unmerge the weights if merged
            if self.merged:
                self.sub_monarch_data()
                self.merged = False
        else:
            # in eval if not merged then merge the weights
            if not self.merged:
                self.add_monarch_data()
                self.merged = True    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Monarch layer.
        """

        original_output = super().forward(x)

        if not self.merged:
            # If in training mode, we need to add the Monarch adjustment
            monarch_name = self.params_with_monarch['weight']
            blkdiag1 = getattr(self, f'{monarch_name}_blkdiag1')
            blkdiag2 = getattr(self, f'{monarch_name}_blkdiag2')

            # Calculate the adjustment M*x
            adjustment = self.monarch_forward(x, blkdiag1, blkdiag2)
            
            # print(f"Applying Monarch adjustment with scaling {self.scaling} to the output.")
            return original_output + (self.scaling * adjustment) 
        else:
            # If in eval mode, we return the original output
            # print("Warning: Monarch layer is in evaluation mode. No Monarch adjustment applied.")
            return original_output

    def preprocess(self, x):
        x_in_features = x.shape[-1]
        if x_in_features < self.num_blocks * self.block_size:
            x = F.pad(x, (0, self.num_blocks * self.block_size - x_in_features))
        return x

    def postprocess(self, output):
        output_out_features = output.shape[-1]
        if output_out_features > self.out_features:
            output = output[..., : self.out_features]
        return output

    def add_monarch_data(self):
        """Add the Monarch adjustment to the original weights."""
        for param_name in self.params_with_monarch:
            # Get the M matrix
            adjustment = self._get_monarch_matrix(param_name)
            
            # Add it to the original weight's data
            base_param = getattr(self, param_name)
            base_param.data += adjustment

    def sub_monarch_data(self):
        """Subtract the Monarch adjustment from the original weights."""
        for param_name in self.params_with_monarch:
            # Get the dense M matrix
            adjustment = self._get_monarch_matrix(param_name)
            
            # Subtract it from the original weight's data
            base_param = getattr(self, param_name)
            base_param.data -= adjustment

    def _get_monarch_matrix(self, param_name: str) -> torch.Tensor:
        """Get the Monarch matrix M for the given parameter name."""

        monarch_name = self.params_with_monarch[param_name]
        blkdiag1 = getattr(self, f'{monarch_name}_blkdiag1')
        blkdiag2 = getattr(self, f'{monarch_name}_blkdiag2')

        identity_matrix = torch.eye(self.in_features, device=self.device, dtype=self.dtype)
        dense_monarch_matrix = self.monarch_forward(identity_matrix, blkdiag1, blkdiag2)

        return dense_monarch_matrix.T
    