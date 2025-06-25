import os

import torch
import torch.nn as nn

from typing import Dict

from morelib.layers import BaseLayer, MonarchLayer
from morelib.more_mha import PlainMultiheadAttentionMoRE
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    'ViT-B/16': {
        'top1': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}

def mark_only_monarch_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        # print(f"Parameter: {n}")
        if '_blkdiag' not in n:
            p.requires_grad = False
            # print(f"Parameter {n} marked as non-trainable.")
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
        return {k: my_state_dict[k] for k in my_state_dict if '_blkdiag' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if '_blkdiag' in k or 'bias' in k}
    elif bias == 'monarch_only':
        to_return = {}
        for k in my_state_dict:
            if 'more_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('_blkdiag')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def get_monarch_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if '_blkdiag' in name:
                params.append(param)
        elif bias == 'all':
            if '_blkdiag' in name or 'bias' in name:
                params.append(param)
        elif bias == 'monarch_only':
            if '_blkdiag' in name:
                params.append(param)
                bias_name = name.split('_blkdiag')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_monarch(args, clip_model):
    list_monarch_layers = []
    # print("entrei na apply_monarch")
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        # print(f"Applying MoRE to {name} in text encoder block {i}")
                        new_multi_head_monarch = PlainMultiheadAttentionMoRE(
                            submodule, enable_monarch=args.params, num_blocks=args.num_blocks, block_rank=args.block_rank, dropout_rate=args.dropout_rate, scaling = args.alpha)
                        new_multi_head_monarch.to(device)
                        setattr(block, name, new_multi_head_monarch)
                        # setattr(submodule, name, new_multi_head_monarch)
                        list_monarch_layers.append(new_multi_head_monarch)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        # print(f"Applying MoRE to {name} in vision encoder block {i}")
                        new_multi_head_monarch = PlainMultiheadAttentionMoRE(
                            submodule, enable_monarch=args.params, num_blocks=args.num_blocks, block_rank=args.block_rank, dropout_rate=args.dropout_rate, scaling = args.alpha)
                        new_multi_head_monarch.to(device)
                        setattr(block, name, new_multi_head_monarch)
                        # setattr(submodule, name, new_multi_head_monarch)
                        list_monarch_layers.append(new_multi_head_monarch)
    return list_monarch_layers

def save_monarch(args, list_monarch_wrappers):
    """
    Saves the trained Monarch adapter weights and metadata to a file.
    
    Args:
        args: Command-line arguments containing metadata.
        list_monarch_wrappers: A list of the PlainMultiheadAttentionMoRE modules.
    """
    weights = {}
    
    for i, mha_wrapper in enumerate(list_monarch_wrappers):
        layer_weights = {}
        if isinstance(mha_wrapper.q_proj, MonarchLayer):
            layer_weights['q_proj'] = {
                'w_blkdiag1': mha_wrapper.q_proj.w_blkdiag1.data,
                'w_blkdiag2': mha_wrapper.q_proj.w_blkdiag2.data
            }
        if isinstance(mha_wrapper.k_proj, MonarchLayer):
            layer_weights['k_proj'] = {
                'w_blkdiag1': mha_wrapper.k_proj.w_blkdiag1.data,
                'w_blkdiag2': mha_wrapper.k_proj.w_blkdiag2.data
            }
        if isinstance(mha_wrapper.v_proj, MonarchLayer):
            layer_weights['v_proj'] = {
                'w_blkdiag1': mha_wrapper.v_proj.w_blkdiag1.data,
                'w_blkdiag2': mha_wrapper.v_proj.w_blkdiag2.data
            }
        if isinstance(mha_wrapper.proj, MonarchLayer): # The output projection
            layer_weights['proj'] = {
                'w_blkdiag1': mha_wrapper.proj.w_blkdiag1.data,
                'w_blkdiag2': mha_wrapper.proj.w_blkdiag2.data
            }
            
        weights[f'layer_{i}'] = layer_weights
        
    metadata = {
        'num_blocks': args.num_blocks,
        'block_rank': args.block_rank,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position,
        'dropout_rate': args.dropout_rate
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # This part for creating the directory and saving is great and can be kept
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/{args.n_iters}iters'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'Monarch weights saved to {save_path}')

def load_monarch(args, list_monarch_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/{args.n_iters}iters/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['num_blocks'] != args.num_blocks:
        raise ValueError(
            f"num_blocks mismatch: expected {args.num_blocks}, found {metadata['num_blocks']}")
    if metadata['block_rank'] != args.block_rank:
        raise ValueError(
            f"block_rank mismatch: expected {args.block_rank}, found {metadata['block_rank']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")
    if metadata['dropout_rate'] != args.dropout_rate:
        raise ValueError(
            f"Position mismatch: expected {args.dropout_rate}, found {metadata['dropout_rate']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_monarch_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_blkdiag1.data.copy_(
                layer_weights['q_proj']['w_blkdiag1'])
            layer.q_proj.w_blkdiag2.data.copy_(
                layer_weights['q_proj']['w_blkdiag2'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_blkdiag1.data.copy_(
                layer_weights['k_proj']['w_blkdiag1'])
            layer.k_proj.w_blkdiag2.data.copy_(
                layer_weights['k_proj']['w_blkdiag2'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_blkdiag1.data.copy_(
                layer_weights['v_proj']['w_blkdiag1'])
            layer.v_proj.w_blkdiag2.data.copy_(
                layer_weights['v_proj']['w_blkdiag2'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_blkdiag1.data.copy_(layer_weights['proj']['w_blkdiag1'])
            layer.proj.w_blkdiag2.data.copy_(layer_weights['proj']['w_blkdiag2'])

    print(f'MoRE weights loaded from {load_path}')