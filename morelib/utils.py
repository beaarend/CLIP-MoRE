import os

import torch
import torch.nn as nn

from typing import Dict

from morelib.layers import BaseLayer, MonarchLayer
from morelib.more_mha import PlainMultiheadAttentionMoRE

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
}


def mark_only_monarch_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'monarch_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'monarch_only':
        for m in model.modules():
            if isinstance(m, MonarchLayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def monarch_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'blkdiag_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'more_' in k or 'bias' in k}
    elif bias == 'monarch_only':
        to_return = {}
        for k in my_state_dict:
            if 'more_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('more_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_monarch_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'more_' in name:
                params.append(param)
        elif bias == 'all':
            if 'more_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'monarch_only':
            if 'more_' in name:
                params.append(param)
                bias_name = name.split('more_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_monarch(args, clip_model):
    list_monarch_layers = []
    print("entrei na apply_monarch")
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.backbone.transformer
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        print(f"Applying MoRE to {name} in text encoder block {i}")
                        new_multi_head_monarch = PlainMultiheadAttentionMoRE(
                            submodule, enable_monarch=args.params, num_blocks=args.num_blocks, block_rank=args.block_rank, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_monarch)
                        list_monarch_layers.append(new_multi_head_monarch)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.backbone.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        print(f"Applying MoRE to {name} in vision encoder block {i}")
                        new_multi_head_monarch = PlainMultiheadAttentionMoRE(
                            submodule, enable_monarch=args.params, num_blocks=args.num_blocks, block_rank=args.block_rank, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_monarch)
                        list_monarch_layers.append(new_multi_head_monarch)
    return list_monarch_layers
