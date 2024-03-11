# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import os
import pdb

import torch
import numpy as np
from functools import partial
from dict_recursive_update import recursive_update
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_train

import torch.distributed as dist

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from core.utils import NestedTensor
from ..ckpt import checkpoint_wrapper

from einops import rearrange, repeat
from torch.distributions.dirichlet import Dirichlet

from typing import Dict, List, Optional, Tuple, Union

import einops
try:
    import xformers
    import xformers.ops
    ATTENTION_MODE = 'xformers'
except:
    ATTENTION_MODE = 'math'
print(f'ATTENTION_MODE: {ATTENTION_MODE}')

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, window_size=None, rel_pos_spatial=False,
            ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.window_size = window_size
        if COMPAT:
            if COMPAT == 2:
                self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_dim))
            else:
                q_size = window_size[0]
                kv_size = q_size
                rel_sp_dim = 2 * q_size - 1
                self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # # origin attn
        if ATTENTION_MODE == 'math':
            attn = ((q * self.scale) @ k.transpose(-2, -1))
            # import pdb;pdb.set_trace()
            if self.rel_pos_spatial:
                raise
                attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
            if mask is not None:
                key_padding_mask = mask.unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(key_padding_mask, float('-inf'))

            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # flash attn
        # x = F.scaled_dot_product_attention(q, k, v).reshape(B, N, C)

        elif ATTENTION_MODE == 'xformers':
            qkv = self.qkv(x)
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)

        x = self.proj(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.

    Source: https://github.com/facebookresearch/mvit/
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial

        if COMPAT:
            q_size = window_size[0]
            kv_size = window_size[1]
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x = x.reshape(B_, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        B_w = x.shape[0]
        N_w = x.shape[1]
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise

        attn = attn.softmax(dim=-1)
        _attn_mask = (torch.isinf(attn) + torch.isnan(attn))
        attn = attn.masked_fill(_attn_mask, 0)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False, prompt=None, ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
                )
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)


    def forward(self, x,  mask=None):
        # import pdb;pdb.set_trace()
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, mask=None, **kwargs):
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        return x, (Hp, Wp), mask


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class maskViT(nn.Module):
    """
    Vision Transformer without adaptor. Tokens from patch adapter and label adapter are randomly masked and fed into
    the model.
    :param img_size: int, size of image. A placeholder to generate self.temp_patch_shape, which is useless in the default
    setting when rel_pos_spatial is False and window is False.
    :param patch_size: int, size of patch. A placeholder to generate self.temp_patch_shape, which is useless in the default
    setting when rel_pos_spatial is False and window is False.
    :param embed_dim: int, dimension of embedding.
    :param depth: int, number of layers.
    :param num_heads: int, number of heads.
    :param mlp_ratio: float, ratio of mlp hidden dim to embedding dim.
    :param qkv_bias: bool, whether to add bias to qkv.
    :param drop_path_rate: float, drop rate in layers.
    :param norm_layer: torch norm module, given norm module or torch.nn.LayerNorm, normalization layer.
    :param window: bool, whether to use window attention. default False.
    :param interval: int, interval of the appearance of window attention, which is useless when window is False.
    :param test_pos_mode: bool, whether to use test position mode. default False.
    :param task_sp_list: list, task specific list for DDP communication. Default: ()
    :param neck_sp_list: list, neck specific list for DDP communication. Default: ()
    :param learnable_pos: bool, whether to use learnable position embedding. default False.
    :param rel_pos_spatial: bool, whether to use relative position embedding. default False.
    :param lms_checkpoint_train: bool, whether to use lms checkpoint training. Always set to True to reduce the memory
    consumption.
    :param prompt: str, whether to use prompt for prompt tuning. default None.
    :param pad_attn_mask: bool, whether to use padding attention mask. default False.
    :param freeze_iters: int, number of iterations to freeze the model. default 0. (not used)
    :param act_layer: str, activation layer. default 'GELU'.
    :param pre_ln: bool, whether to use pre layer norm. default False.
    :param mask_input: bool, whether to mask input. default False.
    :param ending_norm: bool, whether to use layer norm at the end of the model. default True.
    :param round_padding: bool, whether to use round padding. default False.
    :param compat: bool, whether to use compatibility mode. default False.
    :param num_encoded_tokens: int, number of encoded tokens. Now dropped, the number of encoded tokens is now computed
    from the number of patches using vis_patch_token_ratio and vis_label_token_ratio.
    :param mask_ratio_alphas: float, mask ratio alphas for Dirichlet sampling. default 1.0.
    :param sample_tasks_uniformly: bool, whether to sample tasks uniformly. default False.
    :param vis_patch_token_ratio: float, ratio of visible patch tokens to total patch tokens. act when bigger than 0.
    :param vis_label_token_ratio: float, ratio of visible label tokens to total label tokens. act when bigger than 0.
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_path_rate=0., norm_layer=None, window=False,
                 interval=3, test_pos_mode=False, use_abs_pos_emb=True, bn_group=None,
                 task_sp_list=(), neck_sp_list=(), learnable_pos=False, rel_pos_spatial=False,
                 lms_checkpoint_train=False,
                 prompt=None, pad_attn_mask=False, freeze_iters=0,
                 act_layer='GELU', pre_ln=False, mask_input=False, ending_norm=True,
                 round_padding=False, compat=False,
                 num_encoded_tokens=128, mask_ratio_alphas=1.0, sample_tasks_uniformly=False, num_global_tokens=0,
                 mask_all_gt_tokens=False, vis_patch_token_ratio=-1, vis_label_token_ratio=-1, attn_calcul_method='',
                 ):
        super().__init__()
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.task_sp_list = task_sp_list
        self.neck_sp_list = neck_sp_list
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding

        self.num_encoded_tokens = num_encoded_tokens
        self.alphas = mask_ratio_alphas
        self.sample_tasks_uniformly = sample_tasks_uniformly
        self.mask_all_gt_tokens = mask_all_gt_tokens
        self.vis_patch_token_ratio = vis_patch_token_ratio
        self.vis_label_token_ratio = vis_label_token_ratio
        global ATTENTION_MODE
        ATTENTION_MODE = attn_calcul_method if attn_calcul_method else ATTENTION_MODE

        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, embed_dim))
        trunc_normal_(self.global_tokens, std=0.02)

        global COMPAT
        COMPAT = compat

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        #  compute the temporary patch shape for Block
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.temp_patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.blocks = nn.ModuleList()
        # import pdb;pdb.set_trace()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=(14, 14) if ((i + 1) % interval != 0) else self.temp_patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial,
                prompt=prompt,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            if self.lms_checkpoint_train == 'fairscale':
                try:
                    block = checkpoint_wrapper(block)
                    if i==0:
                        print(f'[Rank {dist.get_rank()}] fairscale checkpoint success')
                except:
                    if i==0:
                        print(f'[Rank {dist.get_rank()}] fairscale checkpoint failed, use naive block')
                    pass
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)

        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()
        ###
        self.test_pos_mode = test_pos_mode

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, B, mask=None):
        """
        forward visible features
        :param x: Tensor, [batch size, length, dim]. shuffled input tokens
        :param B: batch size
        :param mask: attn padding mask if exists
        :return: x: encoded features
        """
        batch_size, seq_len, _ = x.size()

        x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity
        for i, blk in enumerate(self.blocks):
            # *Warning*: official ckpt implementation leads to NaN loss in many cases, use fairscale if that's the case
            # lms_checkpoint_train = {False, True, 'fairscale'}
            if self.lms_checkpoint_train == True:
                x = checkpoint_train(lambda x: blk(x, mask), x, preserve_rng_state=True)
            else:
                x = blk(x, mask)

        if self.ending_norm:
            x = self.norm(x)  # b n_vis c

        # x = self.unmasking(x)  # effective only if self.mask_input=True (default False), for mask based ssl

        return x

    def generate_input_info(self, input_task_tokens, ):
        """
        generate input info for other modules
        :param input_task_tokens: dict [str: Tensor], [batch size, length, dim]. shuffled input tokens from different
            modalities.
        :return: input_info: dict. contains input info for other modules
        """
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            if isinstance(tensor, NestedTensor):
                num_tokens = tensor.tensors.shape[1]
            else:
                num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,  # TODO: Modify when adding non-2D tasks
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        # input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens

        return input_info

    def sample_alphas(self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(self,
                            input_tokens: Dict[str, torch.Tensor],
                            num_encoded_tokens: int = 128,
                            alphas: Union[float, List[float]] = 1.0,
                            sample_tasks_uniformly: bool = False) :
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select. Now dropped, number of encoded tokens is now computed by
            the model itself using the given vis_patch_token_ratio and vis_label_token_ratio. TODO: Remove this param
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        #  replace the following with NestedTensor-supported code
        input_tokens_list = []
        for v in input_tokens.values():
            if isinstance(v, NestedTensor):
                input_tokens_list.append(v.tensors)
            else:
                input_tokens_list.append(v)

        B = input_tokens_list[0].shape[0]
        device = input_tokens_list[0].device
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in input_tokens_list]



        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        if self.mask_all_gt_tokens:
            task_sampling_dist[:,0]=1
            task_sampling_dist[:,1]=0
        elif self.vis_patch_token_ratio>=0 and self.vis_label_token_ratio>=0:
            task_sampling_dist[:, 0] = self.vis_patch_token_ratio
            task_sampling_dist[:, 1] = self.vis_label_token_ratio

        task_masks = []


        #  for each task, sample tokens
        samples_per_task = torch.zeros_like(task_sampling_dist, device=task_sampling_dist.device)
        samples_per_task[:, 0] = task_sampling_dist[:, 0] * num_tokens_per_task[0]
        samples_per_task[:, 1] = task_sampling_dist[:, 1] * num_tokens_per_task[1]
        samples_per_task = samples_per_task.round().long()
        # import pdb;pdb.set_trace()
        num_encoded_tokens = samples_per_task[0].sum()

        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)
        # import pdb;pdb.set_trace()
        mask_all = torch.cat(task_masks, dim=1)  #[B, (num_tokens, num_tokens)]
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)}

        return task_masks, ids_keep, ids_restore, ids_shuffle

    def forward(self, input_var):
        """
        get input task tokens from patch and label adapters, then generate random masks for each task.
        concatenate all task tokens and masks, then feed into the encoder.

        :param input_var:
        :return: input_var. updated with backbone_masking_info, backbone_output and backbone_task_info
        """
        output = {}
        # several input tokens from several adapters.


        input_task_tokens = {}
        input_task_infos = {}
        domains = []

        for key in input_var.keys():
            if key.startswith('adapter_output_'):
                domain = key.replace('adapter_output_','')
                domains.append(domain)
                input_task_tokens[domain] = input_var[key]['tokens']
                input_task_infos[domain]={'N_H': input_var[key]['N_H'],
                                          'N_W': input_var[key]['N_W'],
                                          'attn_mask': input_var[key]['attn_mask'],
                                          'Bs':input_var[key]['Bs']}
                B = input_var[key]['Bs']

        # for supervised learning, number of visible tokens (patch tokens, domains[0]) CHECK

        # import pdb;pdb.set_trace()
        # H, W = input_var['image'].shape[-2:]
        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, )
        task_masks, ids_keep, ids_restore, ids_shuffle = self.generate_random_masks(
            input_task_tokens,
            self.num_encoded_tokens,
            alphas=self.alphas,
            sample_tasks_uniformly=self.sample_tasks_uniformly
        )

        # for 10-clip test in sk-action
        if ('text' in input_task_tokens and 'sparse_labeling' in input_task_tokens )\
                and (input_task_tokens['text'].shape[0] != input_task_tokens['sparse_labeling'].shape[0]):
            bt = input_task_tokens['text'].shape[0]
            bp = input_task_tokens['sparse_labeling'].shape[0]
            num_clip = int(bp/bt)
            _,n,d = input_task_tokens['text'].shape
            input_task_tokens['text'] = input_task_tokens['text'].unsqueeze(1).repeat(1,num_clip,1,1).reshape(-1,n,d)
            global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B*num_clip)
        else:
            global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)

        input_tokens = []
        pad_attn_masks = []

        for task_tokens in input_task_tokens.values():
            if isinstance(task_tokens, NestedTensor):
                input_tokens.append(task_tokens.tensors)
                pad_attn_masks.append(task_tokens.mask)
            else:
                input_tokens.append(task_tokens)
                pad_attn_masks.append(torch.zeros([task_tokens.shape[0], task_tokens.shape[1]]
                                                  ).bool().to(task_tokens.device))

        input_tokens = torch.cat(input_tokens, dim=1)
        pad_attn_masks = torch.cat(pad_attn_masks, dim=1)
        # input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)
        output['backbone_masking_info'] ={'pad_attn_masks':pad_attn_masks,}

        # Apply mask and shuffle
        input_tokens = torch.gather(input_tokens, dim=1,
                                    index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))
        pad_attn_masks = torch.gather(pad_attn_masks, dim=1,
                                      index=ids_keep)
        import pdb
        # pdb.set_trace()

        # global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        output['backbone_task_info'] = input_info # for decoder mlp

        # TODO : STILL some bugs when using nested images. Default the else branch
        if self.pad_attn_mask:
            # raise ValueError('not implemented.')
            output['backbone_output'] = self.forward_features(input_tokens, B, mask=pad_attn_masks)
            # import pdb;pdb.set_trace()
            # output['backbone_output'] = NestedTensor(self.forward_features(x, Hp, Wp, B), mask) # STILL some bugs
        else:
            output['backbone_output'] = self.forward_features(input_tokens, B)

        output['backbone_masking_info'].update({'task_masks': task_masks,
                                           'ids_keep': ids_keep,
                                           'ids_restore': ids_restore,
                                           'ids_shuffle': ids_shuffle,
                                           'task_infos': input_info,
                                           'shuffled_pad_attn_masks': pad_attn_masks,})
        input_var.update(output)
        return input_var

def vit_base_patch16_mask(pretrained=False, load_pos_embed=True, **kwargs):
    """
    instantiate a ViT model (without input adapter part) from a pre-trained checkpoint
    :param pretrained: bool, whether to load the pretrained weights
    :param load_pos_embed: bool, whether to load the position embedding weights
    :param kwargs:
    :return: a ViT model
    """
    default = dict(
        drop_path_rate=0.1, use_abs_pos_emb=True,  # as in table 11
        ####
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    recursive_update(default, kwargs)
    model = maskViT(**default)
    # del model.head

    if pretrained:
        script_dir = os.path.dirname(__file__)

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
            if dist.get_rank()==0:
                print(f'[Rank {dist.get_rank()}] load HAP pretrained weights')
        elif pretrained == 'humanbench':
            rel_path = "pretrain_weights/humanbench.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
            if dist.get_rank()==0:
                print(f'[Rank {dist.get_rank()}] load humanbench pretrained weights')
        else:
            rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        # load while interpolates position embedding
        load_checkpoint(model, checkpoint, load_pos_embed, strict=False, logger=dummy_logger)
        del checkpoint

    return model

def vit_large_patch16_mask(pretrained=False, load_pos_embed=True, **kwargs):
    default = dict(
        drop_path_rate=0.5, use_abs_pos_emb=True,  # as in table 11
        ####
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    recursive_update(default, kwargs)
    model = maskViT(**default)

    if pretrained:
        script_dir = os.path.dirname(__file__)

        rel_path = "pretrain_weights/mae_pretrain_vit_large.pth"
        checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']

        # load while interpolates position embedding
        load_checkpoint(model, checkpoint, load_pos_embed, strict=False, logger=dummy_logger)
        del checkpoint

    return model

def vit_base_patch16(pretrained=False, load_pos_embed=True, **kwargs):
    default = dict(
        drop_path_rate=0.1, use_abs_pos_emb=True,  # as in table 11
        ####
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    recursive_update(default, kwargs)
    model = ViT(**default)
    # del model.head

    if pretrained:
        script_dir = os.path.dirname(__file__)

        if pretrained == 'supervised-80ecf9dd':
            rel_path = "pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
        elif pretrained == 'clip':
            rel_path = "pretrain_weights/CLIP-ViT-B-16.pt"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
            # rename & clean loaded keys
            checkpoint = clip_checkpoint_preprocess(checkpoint)
        else:
            rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']

        # load while interpolates position embedding
        load_checkpoint(model, checkpoint, load_pos_embed, strict=False, logger=dummy_logger)
        del checkpoint

    return model


def vit_base_patch16_ema(**kwargs):
    backbone = vit_base_patch16(**kwargs)
    backbone.ema = [vit_base_patch16(**kwargs)]
    backbone.ema[0].mask_input = False
    return backbone


class dummy_logger:
    def info(self, **kwargs):
        print(**kwargs)

    def warning(self, **kwargs):
        print(**kwargs)


def clip_checkpoint_preprocess(checkpoint):
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


def load_checkpoint(model, state_dict, load_pos_embed, strict=False, logger=None):
    """
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    if 'pos_embed' in state_dict:
        # if load_pos_embed:
        #     state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
        #                                                     patch_shape=model.patch_embed.patch_shape,
        #                                                     num_extra_tokens=1)
        # else:
        del state_dict['pos_embed']
        print("checkpoint pos_embed removed as it has been loaded in the adapter")

    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

    load_state_dict(model, state_dict, strict, logger)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
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
    print("finish load")


def interpolate_pos_embed(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    # class_token and dist_token are kept unchanged
    print(f"[rank {dist.get_rank()}] Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(
        0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='bicubic', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (b, h*w, c)
    return new_pos_embed

def interpolate_pos_embed_1d(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = pos_embed_checkpoint.shape[-2]
    # class_token and dist_token are kept unchanged
    print(f"[rank {dist.get_rank()}] Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
    # print(pos_tokens.shape)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='linear', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 1)  # (b, h*w, c)
    return new_pos_embed

def interpolate_pos_embed_with_cls_token(pos_embed_checkpoint, patch_shape, num_extra_tokens):
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


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
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
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

