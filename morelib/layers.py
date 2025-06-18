import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

from morelib.triton import monarch_kernel

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


# linear lora
# change lora_layer to more_layer => basically structured linear here
# class StructuredLinear(nn.Module):

#     def __init__(self, in_features, out_features, bias=None, device=None, dtype=None, **kwargs):
#         """Subclasses should call reset_parameters"""
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         # Subclasses may override {in,out}_features_extended
#         if not hasattr(self, "in_features"):
#             self.in_features = in_features
#         if not hasattr(self, "out_features"):
#             self.out_features = out_features

#         # set bias
#         if bias is None:
#             self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
#         elif isinstance(bias, torch.Tensor):
#             assert bias.shape == (out_features,), f"bias shape {bias.shape} is not (out_features,)"
#             self.bias = nn.Parameter(bias)
#         else:
#             self.register_parameter("bias", None)
# 

class MonarchLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, num_blocks: int, block_rank: int, weights=None):
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.block_rank = block_rank
        self.monarch_impl = monarch_kernel
        self.block_size = int(math.ceil(self.in_features / self.num_blocks))

        # Throw away blocks that are fully padded

        if ((self.num_blocks * self.block_size) > self.in_features):
            print(f"Warning: {self.num_blocks} blocks with size {self.block_size} are larger than in_features {self.in_features}. Adjusting num_blocks.")
            self.num_blocks = (self.in_features + self.block_size - 1) // self.block_size
        elif ((self.num_blocks * self.block_size) < self.in_features):
            print(f"Warning: {self.num_blocks} blocks with size {self.block_size} are smaller than in_features {self.in_features}. Adjusting block_size.")
            self.num_blocks = (self.in_features + self.block_size - 1) // self.block_size

        # dropout, merge, scaler?

        # Init block-diagonal monarch factors -> CAREFUL! Original implementation changed paramters orders (block_rank <=> block_size)
        self.diagonal_block_1 = nn.Parameter(
            torch.zeros(
                self.num_blocks, self.block_rank, self.block_size
            )
        )
        self.diagonal_block_2 = nn.Parameter(
            torch.zeros(
                self.num_blocks, self.block_size, self.block_rank
            )
        )
    
    def monarch_adjustment(self, x: torch.Tensor) -> torch.Tensor:

        monarch_output = self.monarch_impl(
            self.preprocess(x),
            self.diagonal_block_1,
            self.diagonal_block_2,
        )
        return self.postprocess(monarch_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Monarch layer.
        :param x: Input tensor of shape (batch_size, in_features).
        :return: Output tensor of shape (batch_size, out_features).
        """
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)
        monarch_adjustment_value = self.monarch_adjustment(x)
        result = original_output + monarch_adjustment_value
        return result
    
    def train(self, mode: bool = True):
        """
        Set the training mode for the Monarch layer.
        :param mode: If True, set the layer to training mode; otherwise, set it to evaluation mode.
        """
        super().train(mode)
        self.base_train(mode)
    
    # Preprocess and postprocess might not be needed because CLIP dimensions are both fixed and the same value, but still...

    def preprocess(self, x):
        self.in_features = x.shape[-1]
        if self.in_features < self.num_blocks * self.block_size:
            x = F.pad(x, (0, self.num_blocks * self.block_size - self.in_features))
        return x

    def postprocess(self, output):
        out_features = output.shape[-1]
        if out_features > self.out_features:
            output = output[..., : self.out_features]
        return output


       

    
    

# class LoRALayer():
#     def __init__(
#         self, 
#         r: int, 
#         lora_alpha: int, 
#         fan_in_fan_out: bool = False,
#         dropout_rate:float = 0,
#     ):
#         self.r = r
#         self.lora_alpha = lora_alpha
#         self.dropout_rate = dropout_rate
#         if self.r > 0:
#             #self.scaling = self.lora_alpha / self.r
#             self.scaling = self.lora_alpha/math.sqrt(self.r) # 
#         # Mark the weight as unmerged
#         self.merged = False
#         # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         self.fan_in_fan_out = fan_in_fan_out
#         # define params that require LoRA {'param_name': 'lora_name'}
#         self.params_with_lora = {}

#     def register_lora_param(self):
#         r"""Register LoRA matrix"""
#         for param_name, lora_name in self.params_with_lora.items():
#             assert len(eval(f'self.{param_name}').size()) == 2
#             self.register_parameter(f'{lora_name}_lora_A', 
#                 nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
#                 )
#             self.register_parameter(f'{lora_name}_lora_B', 
#                 nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
#                 )
                
#             eval(f'self.{param_name}').requires_grad = False

#     # init monarch matrixes here
#     def init_lora_param(self):
#         for param_name, lora_name in self.params_with_lora.items():
#             if hasattr(self, f'{lora_name}_lora_A'):
#                 # initialize A the same way as the default for nn.Linear and B to zero
#                 nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
#                 nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

#     def transpose(self, w: torch.Tensor):
#         return w.transpose(0, 1) if self.fan_in_fan_out else w

#     # in this case it should be something to make M = P1 L P2 R
#     def merge_BA(self, param_name: str):
#         lora_name = self.params_with_lora[param_name]
#         return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))
    
#     def merge_lora_param(self):
#         r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
#         for param_name, lora_name in self.params_with_lora.items():
#             p = set_param(self, param_name, mode='get')
#             # detach() is very important here
            
#             p_new = p.detach() + self.merge_BA(param_name) * self.scaling
#             set_param(self, param_name, param=p_new, mode='update')

#     def add_lora_data(self):
#         r"""NOT differentiable"""
#         for param_name, lora_name in self.params_with_lora.items():
#             eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling
    
#     def sub_lora_data(self):
#         r"""NOT differentiable"""
#         for param_name, lora_name in self.params_with_lora.items():
#             eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling

#        # why merge? difference between LoRA and MoRE is that LoRA merges the weights?????     
#     def lora_train(self, mode: bool = True):
#         if mode:
#             if self.merged and self.r > 0:
#             # Make sure that the weights are not merged
#                 self.sub_lora_data()
#             self.merged = False
#         else:
#             if not self.merged and self.r > 0:
#             # Merge the weights and mark it
#                 self.add_lora_data()
#             self.merged = True 

# class StructuredLinear(nn.Module):

#     def __init__(self, in_features, out_features, bias=None, device=None, dtype=None, **kwargs):
#         """Subclasses should call reset_parameters"""
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         # Subclasses may override {in,out}_features_extended
#         if not hasattr(self, "in_features"):
#             self.in_features = in_features
#         if not hasattr(self, "out_features"):
#             self.out_features = out_features

#         # set bias
#         if bias is None:
#             self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
#         elif isinstance(bias, torch.Tensor):
#             assert bias.shape == (out_features,), f"bias shape {bias.shape} is not (out_features,)"
#             self.bias = nn.Parameter(bias)
#         else:
#             self.register_parameter("bias", None)

#     def reset_parameters(self) -> None:
#         self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
#         self.reset_parameters_bias()

#     def set_weights_from_dense_init(self, dense_init_fn_):
#         raise NotImplementedError

#     def reset_parameters_bias(self):
#         if self.bias is not None:
#             fan_in = self.bias.shape[-1]
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)

#     @property
#     def saving(self):
#         raise NotImplementedError

#     def convert_to_dense_weight(self):
#         factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
#         dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
#         return dense_weight

#     def preprocess(self, x):
#         in_features = x.shape[-1]
#         if in_features < self.in_features:
#             x = F.pad(x, (0, self.in_features - in_features))
#         return x

#     def postprocess(self, output):
#         out_features = output.shape[-1]
#         if out_features > self.out_features:
#             output = output[..., : self.out_features]
#         return output

#     def forward_matmul(self, x):
#         raise NotImplementedError

#     def forward(self, x):
#         output = self.forward_matmul(x)
#         # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
#         return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output