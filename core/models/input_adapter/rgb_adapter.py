# Base on M-MAE
# https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/input_adapters.py
import os
import pdb
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from core.utils import NestedTensor
from dict_recursive_update import recursive_update
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    generate 2d sin-cos position embedding
    :param: embed_dim: int of the embedding dimension
    :param: grid_size: int of the grid height and width
    :return: pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    generate 2d sin-cos position embedding from a grid
    :param embed_dim: int of the embedding dimension
    :param grid: int of the grid height and width
    :return: pos_embed: [grid_size*grid_size, embed_dim]
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    generate 1d sin-cos position embedding from a grid
    :param: output dimension for each position
    :param: a list of positions to be encoded: size (M,)
    :return: (M, D) positional embedding
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def rgb_adapter(pretrained=False, load_pos_embed=True, **kwargs):
    """
    :param pretrained: whether to load pretrained weights of the ViT adapter
    :param load_pos_embed: whether to load pretrained position embedding
    :param kwargs: kwargs for RGBAdapter
    :return: RGBAdapter
    """
    default = dict(
        img_size=224, use_abs_pos_emb=True,   # as in table 11
        ####
    )
    recursive_update(default, kwargs)
    adapter = RGBAdapter(**default)
    # del model.head

    if pretrained:
        script_dir = os.path.dirname(__file__)
        script_dir = script_dir.replace('input_adapter', 'backbones')

        if pretrained == 'supervised-80ecf9dd':
            rel_path = "pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
        elif pretrained == 'clip':
            rel_path = "pretrain_weights/CLIP-ViT-B-16.pt"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
            # rename & clean loaded keys
            checkpoint = clip_checkpoint_preprocess(checkpoint)
        elif pretrained == 'HAP':
            rel_path = "pretrain_weights/HAP.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        elif pretrained == 'humanbench':
            rel_path = "pretrain_weights/humanbench.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        else:
            rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        # pdb.set_trace()
        # load while interpolates position embedding
        load_checkpoint_adpater(adapter, checkpoint, load_pos_embed, strict=False, logger=dummy_logger)
        del checkpoint

    return adapter


class RGBAdapter(nn.Module):
    """
    Input adapter for RGB images. Uses ViT-style patch embedding. Includes task, positional embeddings and projection.
    :param img_size: Input image size
    :param patch_size: Patch size
    :param in_chans: Number of input channels
    :param embed_dim: Dimension of token embeddings
    :param stride_level: Stride level of input image
    :param pad_attn_mask: Whether to pad attention mask to multiple of stride level
    :param test_pos_mode: Whether to use test-time positional embedding mode. Default: False
    :param use_abs_pos_emb: Whether to use absolute positional embedding. Default: True
    :param learnable_pos: Whether to use learnable positional embedding. An corresponding parameter of use_abs_pos_emb. Default: False
    :param round_padding: Whether to round padding to multiple level image feature. Default: False
    :param task_sp_list: List of task specific list for DDP communication. Default: ()
    :param type_embed: Whether to use type embedding. Default: True
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 stride_level=1,
                 pad_attn_mask=False,
                 test_pos_mode=False,
                 use_abs_pos_emb=True,
                 learnable_pos=False,
                 round_padding=False,
                 task_sp_list = (),
                 modality_share_list = (),
                 type_embed=True,
                 type_embed_zero_init=False,
                 translate_box_gt_in_det=False,
                 ):
        super().__init__()
        if isinstance(img_size, list):
            img_size = img_size
        else:
            img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_level = stride_level
        self.embed_dim = embed_dim
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        self.pad_attn_mask = pad_attn_mask
        self.test_pos_mode = test_pos_mode
        self.round_padding = round_padding
        self.translate_box_gt_in_det = translate_box_gt_in_det

        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list

        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=(self.P_H, self.P_W))

        # freeze positional embedding for reconstruction
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_shape, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            raise

        self.type_embed = nn.Embedding(1, embed_dim) if type_embed else None

        # type embedding should be zero initialized to avoid affecting the pre-trained weights
        if type_embed and type_embed_zero_init:
            self.type_embed.weight.data=torch.zeros(1, embed_dim)

        self.initialize_proj_weights()
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def initialize_proj_weights(self):
        """Initialize the projection weights like nn.Linear (instead of nn.Conv2d)"""
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        """Initialize the weights"""
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _normalization(x, mask=None):
        """Image normalization with ImageNet mean and std"""
        """The mask is used to indicate the valid region of the image
           and the invalid region is set to the mean value of ImageNet
           (To support nested tensor in vit)
         """
        if mask is not None:
            x = x * (~mask)[:, None, :, :] + mask[:, None, :, :] * torch.tensor([123.675, 116.280, 103.530]
                                                                        ).view(1, 3, 1, 1).cuda()
        assert len(x.shape) == 4
        x = x.sub(torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1).cuda()).div(
            torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1).cuda())
        return x

    def forward_proj(self, x, mask=None, **kwargs):
        """Forward projection."""
        B = x.shape[0]
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]  # B, C, H, W

        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        return x, (Hp, Wp), mask, B

    def forward_PE(self, x, Hp, Wp, mask=None):
        """Forward positional encoding."""

        if self.test_pos_mode is False:
            if x.size(1) == self.pos_embed.size(1):
                x = x + self.pos_embed  # BxHWxC
            else:  # take top-left if pos_embed > x's dimension
                x = x + self.pos_embed.reshape(1, self.patch_shape[0],
                                               self.patch_shape[1],
                                               self.pos_embed.size(2))[:, :Hp, :Wp, :].reshape(1, x.size(1),
                                                                                               self.pos_embed.size(2))

        elif self.test_pos_mode == 'interpolate_with_nomask':
            # for pedestrian detection, interpolate the position embedding to the current image size
            assert mask is not None   # mask shape, batch_size, H, W
            batch_size = mask.shape[0]
            not_mask = ~mask
            h_list = not_mask.sum(1)[:,0]
            w_list = not_mask.sum(2)[:,0]
            current_patch_shape = (Hp, Wp)

            paramed_pe_shape = self.patch_shape

            pos_embed = self.pos_embed
            pos_embed = pos_embed.reshape(-1, paramed_pe_shape[0], paramed_pe_shape[1],
                                          self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=current_patch_shape, mode='bicubic',
                                                        align_corners=False)
            pos_embed = pos_embed.repeat(batch_size, 1, 1, 1)
            for i in range(batch_size):
                pos_embed[i, :, :h_list[i], :w_list[i]] = torch.nn.functional.interpolate(
                    pos_embed[i, :, :, :].unsqueeze(0), size=(h_list[i], w_list[i]), mode='bicubic',
                    align_corners=False).squeeze(0)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed
        else:
            raise NotImplementedError

        return x

    def with_type_embed(self, x):
        """add type embedding to the input tensor"""
        batch_size, seq_len, _ = x.size()
        return x + self.type_embed.weight.unsqueeze(0).repeat(batch_size, seq_len, 1)

    def _translate_box_gt_in_det(self, input_var, mask=None):
        if mask is not None:
            not_mask=~mask
            H, W = mask.shape[1], mask.shape[2]
            h_list = not_mask.sum(1)[:,0]
            w_list = not_mask.sum(2)[:,0]
            for idx, (h, w) in enumerate(zip(h_list, w_list)):
                input_var.instances[idx].boxes *= torch.as_tensor([w/W, h/H, w/W, h/H], dtype=torch.float32).cuda()

        return input_var


    def forward(self, input_var):
        """
        :param input_var:
        :return: input_var: "image": tensor of shape [batch_size, 3, H, W]
                            "nested_mask": Mask tensor for the nested tensor
                            "prepad_input_size": [h, w] for sem_seg_postprocess if round_padding is used
                            "adapter_output_rgb": dict of rgb adapter output
        """
        output = {}
        x = input_var['image']
        if isinstance(x, NestedTensor):
            x, mask = x.decompose()
        else:
            mask = None
        if mask is not None and mask.sum()>0 and self.translate_box_gt_in_det:
            input_var = self._translate_box_gt_in_det(input_var, mask)

        # Ignore pre_input padding for test support
        x = self._normalization(x, mask)

        if self.round_padding:
            # pre_input padding for non standard img size support,
            # *** used when test image size varies and not divisible by 32 ***
            stride = self.patch_size
            assert stride[0] == stride[1]
            stride = max(stride[0], self.round_padding)
            output["prepad_input_size"] = [x.shape[-2], x.shape[-1]]  # h, w for sem_seg_postprocess
            target_size = (torch.tensor((x.shape[-1], x.shape[-2])) + (stride - 1)).div(stride, rounding_mode="floor") * stride  # w, h
            padding_size = [  # [l,r,t,b]
                0,
                target_size[0] - x.shape[-1],
                0,
                target_size[1] - x.shape[-2],
            ]
            x = F.pad(x, padding_size, value=0.).contiguous()
            if mask is not None:
                mask = F.pad(mask, padding_size, value=True).contiguous()

        output["image"] = x

        output['nested_mask'] = mask
        if isinstance(input_var['image'], NestedTensor) and self.pad_attn_mask:
            x, (Hp, Wp), mask, B = self.forward_proj(x, mask)
        else:
            x, (Hp, Wp), mask, B = self.forward_proj(x)

        x = self.forward_PE(x, Hp, Wp, mask)

        if self.type_embed is not None:
            x = self.with_type_embed(x)

        # nested tensor if mask is not None
        if mask is not None:
            mask = mask.flatten(1)
            x = NestedTensor(x, mask)

        # adding adapter_output_rgb for adapter training, including: tokens, N_H, N_W, Bs, attn_mask
        output['adapter_output_rgb'] = {'tokens': x,
                                        'N_H': Hp,
                                        'N_W': Wp,
                                        'Bs': B,
                                        'attn_mask':mask} # attention mask


        input_var.update(output)

        return input_var

def interpolate_pos_embed(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    """
    interpolate position embedding from a smaller to a larger size
    :param pos_embed_checkpoint: pretrained position embedding
    :param patch_shape: interpolated position embedding size
    :param num_extra_tokens: number of extra tokens, e.g., class token
    :return: interpolated position embedding
    """
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    # class_token and dist_token are kept unchanged
    print(f"[rank {dist.get_rank()}] Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='bicubic', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (b, h*w, c)
    return new_pos_embed


def interpolate_pos_embed_with_cls_token(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    """
    interpolate position embedding from a smaller to a larger size with a pre-trained checkpoint with class token
    :param pos_embed_checkpoint: pretrained position embedding
    :param patch_shape: interpolated position embedding size
    :param num_extra_tokens: number of extra tokens, e.g., class token
    :return: interpolated position embedding
    """
    posemb_tok, posemb_grid = (
        pos_embed_checkpoint[:, :num_extra_tokens],
        pos_embed_checkpoint[0, num_extra_tokens:],
    )
    gs_old_h, gs_old_w = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=patch_shape, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, patch_shape[0] * patch_shape[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    :param module: Module that receives the state_dict.
    :param state_dict: Weights.
    :param strict: whether to strictly enforce that the keys in :attr:`state_dict` match the keys
    returned by this module's :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    :param logger: Logger to log the error message. If not specified, print function will be used.
    :return: The module itself.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank = dist.get_rank()

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    print("finish load PE")

def load_checkpoint_adpater(model, state_dict, load_pos_embed, strict=False, logger=None):
    """
    Load checkpoint from a file for the adapter module.
    :param model: Module to load checkpoint.
    :param state_dict: Accept local filepath, URL, ``torchvision://xxx``, ``open-mmlab://xxx``.
    :param load_pos_embed: Whether to load position embedding.
    :param strict: Whether to allow different params for the model and checkpoint.
    :param logger: The logger for error message.
    :return: The loaded checkpoint. [dict or OrderedDict].
    """
    import pdb; #pdb.set_trace()
    if 'pos_embed' in state_dict:
        if load_pos_embed:
            state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
                                                            patch_shape=model.patch_shape,
                                                            num_extra_tokens=1)
        else:
            del state_dict['pos_embed']
            print("checkpoint pos_embed removed")
    state_dict['proj.weight'] =state_dict.pop('patch_embed.proj.weight')
    state_dict['proj.bias'] = state_dict.pop('patch_embed.proj.bias')
    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
    # pdb.set_trace()
    load_state_dict(model, state_dict, strict, logger)

def clip_checkpoint_preprocess(checkpoint):
    """
    Preprocess the checkpoint before loading.
    :param checkpoint: 
    :return: checkpoint
    """
    for k in list(checkpoint.keys()):
        if k.startswith('visual'):
            if k in ["visual.proj", "visual.class_embedding"]:
                new_k = k
            elif k.startswith('visual.transformer.resblocks'):
                new_k = k[len("visual.transformer.res"):]
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
                new_k = new_k.replace('out_proj', 'proj')
                new_k = new_k.replace('ln_', 'norm')
                new_k = new_k.replace('c_fc', 'fc1')
                new_k = new_k.replace('c_proj', 'fc2')
            else:
                new_k = k[len("visual."):]
                new_k = new_k.replace('positional_embedding', 'pos_embed')
                new_k = new_k.replace('conv1', 'patch_embed.proj')
                new_k = new_k.replace('ln_post', 'norm')
            checkpoint[new_k] = checkpoint[k]
        del checkpoint[k]
    return checkpoint

class dummy_logger:
    """
    Dummy logger for checkpoint loading.
    """
    def info(self, **kwargs):
        print(**kwargs)

    def warning(self, **kwargs):
        print(**kwargs)
