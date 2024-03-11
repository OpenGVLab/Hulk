# ------------------------------------------------------------------------
# Hulk: A Universal Knowledge Translator for Human-centric Tasks
# Copyright (c) 2024 Shanghai AI Laboratory. All Rights Reserved.
# Licensed under the MIT License, [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
from core.models.ops.utils import c2_xavier_fill
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import os
import sys
from core.data.datasets.images.smpl_data_tools._smpl import SMPL, Mesh
import core.data.datasets.images.smpl_data_tools.config_smpl as smpl_cfg

from core.models.ops.utils import Conv2d
from .position_encoding import PositionEmbeddingSine
from ....backbones.maskvit import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, interpolate_pos_embed, interpolate_pos_embed_1d
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import math
import itertools
from ....ckpt import checkpoint_wrapper
import torch.distributed as dist

from einops import rearrange, repeat


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, # query, bs, C
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_post_deep(self, tgt,
                         tgt_mask: Optional[Tensor] = None,
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt * (3 * self.net_depth) ** 0.25 + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        if self.arch == 'pre_norm':
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        elif self.arch == 'deepnorm':
            return self.forward_post_deep(tgt, tgt_mask,
                                          tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9,):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, # bkb_query, bs, C
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.arch == 'pre_norm':
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        elif self.arch == 'deepnorm':
            raise NotImplementedError
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", arch=False, net_depth=9, requires_grad=True,
                 # ffn_pre_norm=False, zero_init=False,
                 # sparse_init=False, zero_gated=False, zero_norm_weight=False
                 ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth
        self.requires_grad = requires_grad

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'linear' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if self.requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False
            self.linear1.weight.data.zero_()
            self.linear2.weight.data.zero_()
            self.linear1.bias.data.zero_()
            self.linear2.bias.data.zero_()



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_post_deep(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt * (3 * self.net_depth) ** 0.25 + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.arch == 'pre_norm':  # false
            return self.forward_pre(tgt)
        elif self.arch == 'deepnorm':
            return self.forward_post_deep(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def custom_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res


def weights_init(module):
    """ Initialize the weights, copy from CTran"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Hulk_Decoder(nn.Module):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = isinstance(param, torch.nn.parameter.UninitializedParameter)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if not is_param_lazy and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue
                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def __init__(self,
                 in_channels,
                 mask_classification,  # True
                 num_classes: int,
                 hidden_dim: int,
                 num_queries: int,
                 nheads: int,
                 dim_feedforward: int,
                 *,
                 dec_layers: int,
                 pre_norm: bool,  # False
                 mask_dim: int,
                 enforce_input_project: bool,  # False
                 mask_on=True,
                 num_feature_levels=3,
                 reid_cfgs=None,
                 pedattr_cfgs=None,
                 peddet_cfgs=None,
                 cfgs=None,
                 ginfo=None,
                 arch=False,
                 cross_pos_embed="sincos",  # use pos embed in cross attention layers
                 backbone_pose_embed=None,
                 cls_out_dim=None,
                 num_patches=196,
                 patch_size=[16, 16],
                 num_patches_label=196,
                 patch_shape=[14,14],
                 patch_shape_label=[14,14],
                 interpolate_mae_pe=True,
                 no_mask_embed=False,
                 learnable_class_embed=False,
                 adding_per_layer_pe=False,
                 fixed_class_embed=False,
                 fixed_class_embed_cfg={},
                 patch_pos_mode=False,
                 label_pos_mode=False,
                 smpl_attention_mask_flag=False,
                 smpl_mae_pe=False,
                 self_attn_mask_type='full',
                 intermediate_output=False,
                 no_adaptive_MLP=False,
                 text_proj_MLP=False,
                 mask_token_normal_init=False,
                 detach_from_peddet=False,
                 use_adapt_pos2d=False,  # for detection task, adapt pos2d is always used
                 caption_cfgs=None, # for image caption task, used to init word embedding layer
                 use_adapt_pos1d=False,  # for smpl task
                 use_adapt_position='after',  # for smpl task
                 use_smpl_label_attention_mask = False,
                 lms_checkpoint_train=False,
                 ):
        """
        An MAE-like decoder with only self-attention layers and FFN layers. The encoded features, together with the
        corresponding type embeddings and position embeddings, are fed into the decoder to generate the output sequence.
        :param in_channels: placeholder, the number of channels of the input feature map. TODO: remove this parameter
        :param mask_classification: bool, whether to perform mask classification
        :param num_classes: int, the number of classes for mask classification
        :param hidden_dim: int, the hidden dimension of the transformer
        :param num_queries: int, the number of queries for mask classification. only used when learnable_class_embed is
                            True, to generate the learnable class embeddings.
        :param nheads: int, the number of heads in the transformer
        :param dim_feedforward: int, the hidden dimension of the FFN layers
        :param dec_layers: int, the number of transformer layers
        :param pre_norm: bool, whether to perform pre-norm in the transformer
        :param mask_dim: int, the dimension of the mask embeddings
        :param enforce_input_project: palceholder, whether to project the input features to the hidden dimension
        :param mask_on: placehoder, whether to perform mask classification. TODO: remove this parameter
        :param num_feature_levels: int, the number of feature levels, NOT used in this decoder
        :param reid_cfgs: dict, placeholder, the configuration for the reid head. Not used.
        :param pedattr_cfgs: dict, placeholder, the configuration for the pedattr head. Not used.
        :param peddet_cfgs: dict, placeholder, the configuration for the peddet head. Not used.
        :param cfgs: dict, placeholder, the configuration for the decoder. Not used.
        :param ginfo: dict. the global information.
        :param arch: str, the architecture of the model. always use fanin in UniHCPv2
        :param cross_pos_embed: str, the type of positional embedding in the cross attention layers. Not used in UniHCPv2
        :param backbone_pose_embed: str, the type of positional embedding in the backbone. Not used in UniHCPv2
        :param cls_out_dim: int, placeholder. A part of reid_cfgs. Not used in UniHCPv2
        :param num_patches: int, the number of patches in the patch, duplicated option with patch_shape.
                        num_patches = patch_shape[0] * patch_shape[1]
        :param patch_size: list, the size of the patch. default: [16, 16]
        :param num_patches_label: int, the number of patches in the label, duplicated option with patch_shape_label.
                        num_patches_label = patch_shape_label[0] * patch_shape_label[1]
        :param patch_shape: list, the shape of the patch. default: [14, 14]
        :param patch_shape_label: list, the shape of the label. default: [14, 14]
        :param interpolate_mae_pe: bool, whether to interpolate the MAE positional embedding (fix PE) to the patch size.
        :param no_mask_embed: bool, whether to use mask embedding in the decoder. When True, only use patch embedding to
                            reconstruct the label patch. When False, use both mask embedding and patch embedding.
        :param learnable_class_embed: bool, whether to use learnable class embedding. When True, use learnable class
                            as additional input to the decoder. When False, use the class embedding from the BERT as the
                            dynamic convolutional kernel.
        :param adding_per_layer_pe: bool, whether to add positional embedding in each layer.
        :param patch_pos_mode: bool/str, special cases [interpolate from a fix shaped patch branch P.E. {2d detection-sup},
                                                        use human joint(s) position to init P.E. {skeleton action-two branch}]
        :param label_pos_mode: bool/str, special cases [use anchor position(3d points) to be the P.E, {2d detection-sup}]
        :param self_attn_mask_type: str, type to generate self attn masks
        :param intermediate_output: bool, whether to output the intermediate features in the decoder.
        :param no_adaptive_MLP: bool, whether to use adaptive MLP to project the P.E. in the decoder.
        :param text_proj_MLP: bool, whether to use MLP to project the text embedding in the decoder.
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model, duplicated option"
        self.mask_on = mask_on
        self.cross_pos_embed = cross_pos_embed
        self.backbone_pose_embed = [backbone_pose_embed]

        self.reid_cfgs = {} if reid_cfgs is None else reid_cfgs
        self.pedattr_cfgs = {} if pedattr_cfgs is None else pedattr_cfgs
        self.peddet_cfgs = {} if peddet_cfgs is None else peddet_cfgs
        self.caption_cfgs = {} if caption_cfgs is None else caption_cfgs
        self.cfgs = {} if cfgs is None else cfgs
        self.no_mask_embed = no_mask_embed
        self.learnable_class_embed = learnable_class_embed
        self.adding_per_layer_pe = adding_per_layer_pe
        self.fixed_class_embed = fixed_class_embed
        self.fixed_class_embed_cfg = fixed_class_embed_cfg
        self.mask_token_normal_init = mask_token_normal_init
        self.use_adapt_pos2d = use_adapt_pos2d
        self.use_adapt_pos1d = use_adapt_pos1d
        self.use_adapt_position = use_adapt_position

        self.patch_pos_mode = patch_pos_mode
        self.label_pos_mode = label_pos_mode
        self.self_attn_mask_type = self_attn_mask_type
        self.intermediate_output = intermediate_output
        self.no_adaptive_MLP = no_adaptive_MLP
        self.text_proj_MLP = text_proj_MLP
        self.detach_from_peddet = detach_from_peddet
        self.lms_checkpoint_train = lms_checkpoint_train

        # define Transformer decoder here
        self.hidden_dim = hidden_dim
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.query_embed_label_xy = None
        self.query_embed_label_z = None
        self.smpl_attention_mask = None  # for smpl
        self.smpl_attention_mask_flag = smpl_attention_mask_flag
        self.smpl_mae_pe = smpl_mae_pe
        self.use_smpl_label_attention_mask = use_smpl_label_attention_mask
        for _ in range(self.num_layers):

            selfattnlayer = SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )
            if self.lms_checkpoint_train == 'fairscale':
                try:
                    selfattnlayer = checkpoint_wrapper(selfattnlayer)
                    if _==0:
                        print(f'[Rank {dist.get_rank()}] fairscale checkpoint success')
                except:
                    if _==0:
                        print(f'[Rank {dist.get_rank()}] fairscale checkpoint failed, use naive block')
                    pass

            self.transformer_self_attention_layers.append(
                selfattnlayer
            )

            ffnlayer = FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    arch='pre_norm' if pre_norm else arch,
                )

            if self.lms_checkpoint_train == 'fairscale':
                try:
                    ffnlayer = checkpoint_wrapper(ffnlayer)
                    if _==0:
                        print(f'[Rank {dist.get_rank()}] fairscale checkpoint success')
                except:
                    if _==0:
                        print(f'[Rank {dist.get_rank()}] fairscale checkpoint failed, use naive block')
                    pass

            self.transformer_ffn_layers.append(
                ffnlayer
            )


        # decoder specifics
        if self.caption_cfgs != {}:
            if self.caption_cfgs.get('nn.parameter', False):
                num_mask_tokens = self.caption_cfgs.get('vocal_size', 30522)

                if self.caption_cfgs.get('bert_feats_for_embedding', False):
                    self.mask_token = nn.Parameter(torch.zeros(1, num_mask_tokens, 768))  # useless
                    self.register_buffer('mask_token_buffer', torch.zeros(1, num_mask_tokens, 768))
                    self.mask_token_proj = nn.Linear(768, hidden_dim)
                    self.mask_token_buffer.copy_(torch.load('caption_bert.pth', 'cpu').unsqueeze(0))
                else:
                    self.mask_token = nn.Parameter(torch.zeros(1, num_mask_tokens, hidden_dim))
                    if self.mask_token_normal_init:
                        nn.init.normal_(self.mask_token)
            else:
                # self.word_embedding = WordEmbedding(**self.caption_cfgs)
                raise "default setting now not using wordembedding anymore!!!"
            # init ln and dropout
            if self.caption_cfgs.get('lndo', False):
                self.captiontoken_ln = nn.LayerNorm(self.hidden_dim, eps=1e-12)
                self.captiontoken_dropout = nn.Dropout(p=0.1)

        else:
            num_mask_tokens = self.peddet_cfgs.get('share_content_query', 1)
            self.mask_token = nn.Parameter(torch.zeros(1, num_mask_tokens, hidden_dim))
            if self.mask_token_normal_init:
                nn.init.normal_(self.mask_token)
        ### in the ||detection|| task, a fix anchor point are projected into 3d p.e.
        ### and use adaptive mlp to be the real p.e.
        if self.peddet_cfgs.get('num_queries',0):
            num_anchors = self.peddet_cfgs.get('num_queries') // self.peddet_cfgs.get('share_content_query', 1)
            self.anchor = nn.Parameter(torch.zeros(1, num_anchors, self.peddet_cfgs.get('query_pe_dim', 3),),
                                       requires_grad=self.peddet_cfgs.get('anchor_requires_grad', True))  # xyz, 3 dim
            assert self.peddet_cfgs.get('pre_defined_path', '')
            anchor_points = np.load(self.peddet_cfgs.get('pre_defined_path'))
            self.anchor.data.copy_(torch.from_numpy(anchor_points))
        else:
            self.anchor = None

        #  patch and label task token are given in the unihcpv2.py, which uses the parameters defined in the neck to
        #  represent the modalities in the decoder
        self.patch_task_token = None
        self.label_task_token = None

        self.decoder_norm = nn.LayerNorm(hidden_dim) if self.pedattr_cfgs.get('head_nrom_type', False) != 'post' else nn.LayerNorm(hidden_dim*num_queries)

        self.num_queries = num_queries



        # fix absolute P.E. for masking decoder
        # self.query_embed(_label) for patch(label) positional embedding
        self.patch_shape_patch = patch_shape
        self.patch_shape_label = patch_shape_label
        assert num_patches == self.patch_shape_patch[0] * self.patch_shape_patch[1], "num patches from the adapter(patch branch) " \
                                                                         "should equal to patch_shape[0] * patch_shape[1]"
        self.query_embed_patch = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        self.query_embed_label = nn.Parameter(torch.zeros(1, patch_shape_label[0] * patch_shape_label[1], hidden_dim),
                                              requires_grad=False)



        if patch_shape[0]>1:
            if interpolate_mae_pe:
                # for 2D image, we can interpolate the positional embedding from a pre-define one
                decoder_pos_embed = get_2d_sincos_pos_embed(self.query_embed_patch.shape[-1], int(224 ** .5))
                decoder_pos_embed = interpolate_pos_embed(pos_embed_checkpoint=torch.from_numpy(decoder_pos_embed), patch_shape=[patch_shape[0], patch_shape[1]],
                                                          num_extra_tokens=0)
            else:
                decoder_pos_embed = get_2d_sincos_pos_embed(self.query_embed_patch.shape[-1],
                                                            grid_size=[patch_shape[0], patch_shape[1]])
        elif patch_shape[0]==1:
            # position 1d should be pre-defined, default as pos = [0, 1, 2, 3, 4, 5, 6]  # M = 7
            pos = np.array(range(patch_shape[-1]))
            decoder_pos_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.query_embed_patch.shape[-1],
                                                                  pos=pos))

        self.query_embed_patch.data.copy_(decoder_pos_embed.float())

        # same on the label patch
        if patch_shape_label[0]>1:
            if interpolate_mae_pe:

                decoder_pos_embed_label = get_2d_sincos_pos_embed(self.query_embed_label.shape[-1], int(224 ** 0.5))
                decoder_pos_embed_label = interpolate_pos_embed(pos_embed_checkpoint=torch.from_numpy(decoder_pos_embed_label),
                                                                patch_shape=[patch_shape_label[0], patch_shape_label[1]],
                                                                num_extra_tokens=0)
            else:
                decoder_pos_embed_label = get_2d_sincos_pos_embed(self.query_embed_label.shape[-1],
                                                                  grid_size=[patch_shape[0], patch_shape[1]])
        elif patch_shape_label[0]==1:
            # position 1d should be pre-defined, default as pos = [0, 1, 2, 3, 4, 5, 6]  # M = 7
            pos = np.array(range(patch_shape_label[-1]))
            decoder_pos_embed_label = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.query_embed_label.shape[-1],
                                                                  pos=pos))
        self.query_embed_label.data.copy_(decoder_pos_embed_label.float())

        # level embedding (originally 3 scales)
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # output FFNs
        if self.reid_cfgs and self.reid_cfgs.head_type == 'cat_proj' or self.pedattr_cfgs and self.pedattr_cfgs.head_type == 'cat_proj':
            self.class_embed = nn.Linear(hidden_dim * num_queries, num_classes + 1 if cls_out_dim is None else cls_out_dim)
            self.fc_bias = None
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1 if cls_out_dim is None else cls_out_dim)
            self.fc_bias = None
        try:
            torch.nn.init.constant_(self.fc_bias.weight, 0)
            torch.nn.init.constant_(self.fc_bias.bias, 0)
        except:
            pass
        self.mask_embed = MLP(hidden_dim,
                              hidden_dim,
                              mask_dim,
                              3) if mask_dim and self.cfgs.get('mask_head_type', 'default') == 'default' else None


        self.adapt_pos2d = None


        if self.peddet_cfgs.get('query_pe_dim',0)>0:
        ###  detection
            self.use_adapt_pos2d = True

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ) if self.cross_pos_embed == 'anchor' else None

        self.adapt_pos1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ) if self.cross_pos_embed == 'anchor' else None

        self.pos_embed = None  # valid when cross_pos_embed == 'pos_prior'
        self._reset_parameters()


    def _reset_parameters(self):
        if self.cross_pos_embed == 'pos_prior':
            resolution = self.peddet_cfgs.get('pos_prior_resolution', 224)
            pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, resolution, cls_token=False)  # HW x C
            self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, resolution, resolution),
                                          requires_grad=self.peddet_cfgs.get('pos_prior_embed_update', True))  # BxCxHxW
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().reshape(1, resolution, resolution,
                                                                                  self.hidden_dim).permute(0, 3, 1, 2))


    def get_vis_token_pos_embed(self, shape=None, batch=1):
        if not self.cross_pos_embed:
            return None
        elif self.cross_pos_embed == 'pos_prior':
            pos = F.interpolate(self.pos_embed, size=shape, mode='bicubic',
                                align_corners=False).permute(2, 3, 0, 1).flatten(0, 1)
            return pos
        elif self.cross_pos_embed == 'shared':
            if shape is not None and self.peddet_cfgs != {}:
                #  assume squared initial pose_embed if interpolation is needed and in simple_interpolate mode
                H = W = int(math.sqrt(self.backbone_pose_embed[0].size(1)))
                H_n, W_n = shape
                init_pe = self.backbone_pose_embed[0].reshape(1, H, W, -1)
                pos = F.interpolate(init_pe.permute(0, 3, 1, 2)[None],
                                    size=(self.hidden_dim, H_n, W_n), mode='trilinear', align_corners=False)[0].permute(2, 3, 0, 1).flatten(0, 1)
                return pos
            if len(self.input_proj) > 0:
                pos = self.input_proj[0](self.backbone_pose_embed[0].permute(0, 2, 1).unsqueeze(3)) # b c hw 1
                return pos.squeeze(3).permute(2, 0, 1)  # -> HWxBxC
            return self.backbone_pose_embed[0].permute(1, 0, 2)  # BxHWxC -> HWxBxC
        elif self.cross_pos_embed == 'anchor':
            assert shape is not None
            mask = torch.zeros(batch, *shape, dtype=torch.bool).cuda()
            H_n, W_n = shape
            pos_col, pos_row = mask2pos(mask)  # (1xh, 1xw) workaround to utilize existing codebase
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, H_n, 1).unsqueeze(-1),
                                pos_col.unsqueeze(2).repeat(1, 1, W_n).unsqueeze(-1)], dim=-1)  # 1xhxwx2
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d, self.hidden_dim // 2))  # 1xhxwxc
            return posemb_2d.flatten(1,2).permute(1, 0, 2)  # BxHWxC -> HWxBxC
        elif self.cross_pos_embed == 'shared_inter':
            return F.interpolate(self.backbone_pose_embed[0],
                                 size=self.hidden_dim, mode='linear', align_corners=False).permute(1, 0, 2)
        else:
            raise NotImplementedError(f"unknown self.cross_pos_embed: {self.cross_pos_embed}")

    def forward(self, x, mask_features, mask_label=None):
        pass
    
    def prepare_smpl_PE(self):
        # Generate T-pose template mesh
        smpl = SMPL().to('cuda')
        mesh_sampler = Mesh()
        template_pose = torch.zeros((1,72))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda('cuda')
        template_betas = torch.zeros((1,10)).cuda('cuda')
        template_vertices = smpl(template_pose, template_betas)
 
        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices) #[1, 1723, 3]
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2) #[1, 431, 3]

        # template mesh-to-joint regression 
        template_3d_joints = smpl.get_h36m_joints(template_vertices)
        template_pelvis = template_3d_joints[:,smpl_cfg.H36M_J17_NAME.index('Pelvis'),:]
        template_3d_joints = template_3d_joints[:,smpl_cfg.H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]
        
        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1) #[1, 445, 3]

        ref_vertices_clone = torch.zeros((ref_vertices.shape)).cuda('cuda')
        ref_vertices_clone[:,:,0] = (ref_vertices[:,:,0] + 1) / 2
        ref_vertices_clone[:,:,1] = (ref_vertices[:,:,1] + 1) / 2
        ref_vertices_clone[:,:,2] = (ref_vertices[:,:,2] + 0.2) / 0.4


        grid_embed_label_xy = get_2d_sincos_pos_embed(self.query_embed_patch.shape[-1], int(224 ** 0.5))
        grid_embed_label_xy = interpolate_pos_embed(pos_embed_checkpoint=torch.from_numpy(grid_embed_label_xy),
                                                        patch_shape=[100, 100],
                                                        num_extra_tokens=0).cuda()
        
        ref_vertices_clone = (ref_vertices_clone*100)# we get the pixel coordinate
        xy_index = (ref_vertices_clone[:,:,0].long() + ref_vertices_clone[:,:,1].long() * 100).reshape([-1]).long()
        # import pdb;pdb.set_trace()
        self.query_embed_label_xy = grid_embed_label_xy[:,xy_index].float().cuda()

        grid_embed_label_z = get_1d_sincos_pos_embed_from_grid(self.query_embed_patch.shape[-1], pos = np.arange(int(224 ** 0.5), dtype=np.float32))
        grid_embed_label_z = interpolate_pos_embed_1d(pos_embed_checkpoint=torch.from_numpy(grid_embed_label_z),
                                                        patch_shape=100,
                                                        num_extra_tokens=0).cuda()
        z_index = ref_vertices_clone[:,:,2].reshape([-1]).reshape([-1]).long()
        self.query_embed_label_z = grid_embed_label_z[:,z_index].float().cuda()

        return

    def forward_recons(self, x):
        # x is a dict dict_keys(['filename', 'width', 'height', 'image', 'label', 'instances', 'nested_mask',
        # 'adapter_output_rgb', 'adapter_output_dense_labeling',
        # 'backbone_task_info', 'backbone_output', 'backbone_masking_info', 'neck_output'])
        # re-shuffle info of the random encoded features

        task_infos = x.backbone_masking_info.task_infos
        ids_restore = x.backbone_masking_info.ids_restore
        ids_keep = x.backbone_masking_info.ids_keep
        ids_shuffle = x.backbone_masking_info.ids_shuffle
        idx2modality = {idx: modality for idx, modality in enumerate(task_infos.tasks.keys())}

        patch_modality = idx2modality[0]
        label_modality = idx2modality[1]


        # only patch_features projected by the shared neck are effective
        patch_features = x[f'neck_output_{patch_modality}']['mask_features']
        label_features = x[f'neck_output_{label_modality}']['mask_features']

        batch_size = patch_features.shape[0]

        if 'num_global_tokens' in task_infos and task_infos['num_global_tokens']!=0:
            num_global_tokens = task_infos['num_global_tokens']
            context_tokens_without_global_patch = patch_features[:, :-num_global_tokens]
            context_tokens_without_global_label = label_features[:, :-num_global_tokens]
        else:
            context_tokens_without_global_patch = patch_features
            context_tokens_without_global_label = label_features

        # add mask tokens as the modality tokens,
        if self.caption_cfgs != {}:
            if self.training:
                token_ids, token_padding_mask = x['input_id'][:, :-1], x['padding_mask'][:, :-1]
                token_padding_mask = (1. - token_padding_mask).cpu()
                if self.caption_cfgs.get('bert_feats_for_embedding', False):
                    projed_mask_tokens = self.mask_token_proj(self.mask_token_buffer)
                    mask_tokens_select = projed_mask_tokens[0][token_ids]
                else:
                    mask_tokens_select = self.mask_token[0][token_ids]

                total_length = ids_shuffle.shape[1]
                mask_length = mask_tokens_select.shape[1]

                # use the context_tokens_without_global_patch to replace the fake_image_tokens
                total_tokens_tmp = torch.cat([context_tokens_without_global_patch, mask_tokens_select], dim=1)
                shuffled_tokens_tmp = torch.gather(total_tokens_tmp, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, mask_tokens_select.shape[2]))
                mask_tokens = shuffled_tokens_tmp[:, -mask_length:, :]

            else:
                token_ids, cur_len = x['input_id'], x['cur_len']
                if self.caption_cfgs.get('bert_feats_for_embedding', False):
                    projed_mask_tokens = self.mask_token_proj(self.mask_token_buffer)
                    mask_tokens = torch.index_select(projed_mask_tokens, 1, token_ids.flatten()).squeeze(0).reshape(batch_size, token_ids.shape[1], -1)
                else:
                    mask_tokens = torch.index_select(self.mask_token, 1, token_ids.flatten()).squeeze(0).reshape(batch_size, token_ids.shape[1], -1)
                token_padding_mask = torch.ones((token_ids.shape[0], token_ids.shape[1]))
                token_padding_mask[:, :cur_len] = 0
                total_length = ids_shuffle.shape[1]
                mask_length = mask_tokens.shape[1]
                mask_tokens = torch.gather(torch.cat([torch.zeros(batch_size, total_length-mask_length, mask_tokens.shape[2]).cuda(), mask_tokens], dim=1),
                                        dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, mask_tokens.shape[2]))[:, -mask_length:, :]

        elif self.peddet_cfgs.get('share_content_query', 1) == 1:
            mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=batch_size, n=task_infos['num_task_tokens'] - context_tokens_without_global_patch.shape[1])
        else:

            num_anchors = self.peddet_cfgs.get('num_queries') // self.peddet_cfgs.get('share_content_query', 1)
            mask_tokens = self.mask_token.permute(1, 0, 2).unsqueeze(1).repeat(1, num_anchors, batch_size, 1).\
                reshape(-1, batch_size, self.hidden_dim).permute(1,0,2)  # b n d
            # when mask_tokens have multiple embeddings/parameters, before gather to reshuffle,
            # we need to shuffle the mask_tokens to make the gather correct
            # only support supervised learning, when the masking token are appended after visible  tokens
            total_length = ids_shuffle.shape[1]
            mask_length = mask_tokens.shape[1]
            mask_tokens = torch.gather(torch.cat([torch.zeros(batch_size, total_length-mask_length, mask_tokens.shape[2]).cuda(), mask_tokens], dim=1),
                                       dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, mask_tokens.shape[2]))[:, -mask_length:, :]


        context_with_mask_patch = torch.cat([context_tokens_without_global_patch, mask_tokens], dim=1)
        context_with_mask_label = torch.cat([context_tokens_without_global_label, mask_tokens], dim=1)

        # reshuffle context_with_mask
        context_with_mask_p = torch.gather(context_with_mask_patch, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask_patch.shape[2]))
        context_with_mask_l = torch.gather(context_with_mask_label, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask_label.shape[2]))

        context_with_mask = []
        for task, info in task_infos['tasks'].items():
            if task == idx2modality[0]:
                context_with_mask.append(context_with_mask_p[:, info['start_idx']:info['end_idx'],:])
                patch_length = info['end_idx'] - info['start_idx']
            if task == idx2modality[1]:
                label_length = info['end_idx'] - info['start_idx']
                # apply caption token ln and dropout
                if self.caption_cfgs.get('lndo', False):
                    caption_token = context_with_mask_l[:, info['start_idx']:info['end_idx'], :]
                    caption_token = self.captiontoken_ln(caption_token)
                    caption_token = self.captiontoken_dropout(caption_token)
                    context_with_mask.append(caption_token)
                else:
                    context_with_mask.append(context_with_mask_l[:, info['start_idx']:info['end_idx'], :])

        context_with_mask = torch.cat(context_with_mask, dim=1)

        # generate type/task/modality embedding
        self.task_embeddings = {}

        # find the modality embedding
        self.task_embeddings[idx2modality[0]] = self.patch_task_token
        self.task_embeddings[idx2modality[1]] = self.label_task_token

        task_embeddings = []
        for task, info in task_infos['tasks'].items():
            if self.task_embeddings[task] is not None:
                task_embed = repeat(self.task_embeddings[task].weight.unsqueeze(0), '() () d -> b n d', b=batch_size, n=info['num_tokens'])
            else:
                # zero embedding to indicate no embedding for different modalities
                task_embed = torch.zeros([batch_size, 0, self.hidden_dim], device=self.query_embed_patch.device)
            task_embeddings.append(task_embed)


        task_embeddings = torch.cat(task_embeddings, dim=1)

        if task_embeddings.shape[1]!=0:
            context_with_mask = context_with_mask + task_embeddings

        # generate positional embeddings
        self.positional_embeddings = {}

        self.positional_embeddings[patch_modality] = self.query_embed_patch
        self.positional_embeddings[label_modality] = self.query_embed_label


        padding_attn_masks = x.backbone_masking_info.pad_attn_masks

        current_patch_token_shape = (x[f'adapter_output_{patch_modality}'].N_H, x[f'adapter_output_{patch_modality}'].N_W)

        patch_padding_attn_masks = padding_attn_masks[:,
                                       :current_patch_token_shape[0] * current_patch_token_shape[1]].reshape(
            padding_attn_masks.shape[0], current_patch_token_shape[0], current_patch_token_shape[1])

        patch_positional_embedding = self.generate_pos_with_pos_mode(batch_size, self.query_embed_patch,
                                                                     self.patch_shape_patch, current_patch_token_shape,
                                                                     mode=self.patch_pos_mode,
                                                                     padding_attn_masks=patch_padding_attn_masks)

        current_label_token_shape = (x[f'adapter_output_{label_modality}'].N_H, x[f'adapter_output_{label_modality}'].N_W)
        label_padding_attn_masks = padding_attn_masks[:,
                                        current_patch_token_shape[0] * current_patch_token_shape[1]:].reshape(
            padding_attn_masks.shape[0], current_label_token_shape[0], current_label_token_shape[1])

        # use the peddet_cfgs != {} to indicate whether this is 2d detection task.
        # the difference mainly lies in the label_positional_embedding
        # for 2d det, the label_positional_embedding is generated with num_anchors=sqrt(num_queries / share_content_query)
        if len(self.peddet_cfgs):
            # in 2d detection, the label branch is sparse_labeling, which has the shape of [time_seq, points(2d/3d)]
            # to detect pedestrian, the p.e. of points-dim should align with image p.e.
            anchor_size = int((self.peddet_cfgs.get('num_queries') // self.peddet_cfgs.get('share_content_query', 1)) ** 0.5)
            label_positional_embedding = self.generate_pos_with_pos_mode(batch_size, self.query_embed_patch,
                                                                         self.patch_shape_patch, (anchor_size, anchor_size),
                                                                         mode=self.label_pos_mode,
                                                                         padding_attn_masks=label_padding_attn_masks)
            label_positional_embedding = label_positional_embedding.repeat(1,
                                                                           self.peddet_cfgs.get('share_content_query', 1),
                                                                           1)
            # reference points are used to generate the p.e. of points-dim

            # in 2d detection, reference points are used to generate the p.e. of points-dim
            # reference points are the center of each anchor
            reference = inverse_sigmoid(self.anchor.repeat(batch_size,
                                                           self.peddet_cfgs.get('share_content_query', 1),
                                                           1))
        else:
            # for other tasks, label_positional_embedding is generated with input patch shape (defined in the label_adapter)
            label_positional_embedding = self.generate_pos_with_pos_mode(batch_size, self.query_embed_label,
                                                                         self.patch_shape_label,
                                                                         current_label_token_shape,
                                                                         mode=self.label_pos_mode,
                                                                         padding_attn_masks=label_padding_attn_masks)
            reference = None


        if self.use_adapt_pos2d:
            patch_positional_embedding = self.adapt_pos2d(patch_positional_embedding)
            label_positional_embedding = self.adapt_pos2d(label_positional_embedding)

        positional_embeddings = [patch_positional_embedding, label_positional_embedding]

        positional_embeddings = torch.cat(positional_embeddings, dim=1)   # b n d

        # change the p.e and context_with_mask to the shape of [num_tokens, batch_size, hidden_dim]
        positional_embeddings = positional_embeddings.permute(1, 0, 2)
        context_with_mask = context_with_mask.permute(1, 0, 2)
        if reference is not None:
            reference = reference.permute(1, 0, 2)

        extra_text_num = 0

        if self.adding_per_layer_pe:
            query_pos = positional_embeddings # num_tokens, bs, dim
        else:
            # if pe is not added per layer, then the pe is added before the first layer
            query_pos = None
            context_with_mask = context_with_mask + positional_embeddings

        #  define the name of output for cascade transformer.
        output = context_with_mask

        #  get the pad_attn_masks from backbone_masking_info for nested tensor when 2d detection
        self_attn_mask = self.generate_self_attn_mask(patch_length=patch_positional_embedding.shape[1],
                                                      label_length=label_positional_embedding.shape[1],
                                                      self_attn_mask_type=self.self_attn_mask_type,
                                                      nested_padding_mask=padding_attn_masks,
                                                      extra_text_num=extra_text_num).to(context_with_mask.device)
        if self.use_smpl_label_attention_mask:
            # attention mask
            num_joints = label_length-431-1
            if self.smpl_attention_mask is None:
                
                zeros_1 = torch.tensor(np.zeros((431, num_joints)).astype(bool)) 
                zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + 431))).astype(bool))
                dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

                folder_path = os.path.join(dirname,'core/data/datasets/images/smpl_data_tools/smpl_modeling/')
                adjacency_indices = torch.load(os.path.join(dirname,'core/data/datasets/images/smpl_data_tools/smpl_modeling/data/smpl_431_adjmat_indices.pt'))
                adjacency_matrix_value = torch.load(os.path.join(dirname,'core/data/datasets/images/smpl_data_tools/smpl_modeling/data/smpl_431_adjmat_values.pt'))
                adjacency_matrix_size = torch.load(os.path.join(dirname,'core/data/datasets/images/smpl_data_tools/smpl_modeling/data/smpl_431_adjmat_size.pt'))
                adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()
                temp_mask_1 = (adjacency_matrix == 0)
                temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
                self.smpl_attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
            if self.smpl_attention_mask_flag:
                attention_mask = self.smpl_attention_mask.to(context_with_mask.device)  # joint attention mask
                self_attn_mask[-(num_joints + 431):,-(num_joints + 431):] = attention_mask
        
        if self.caption_cfgs != {}:
            self_attn_padding_mask = self.generate_self_attn_padding_mask(patch_length=patch_positional_embedding.shape[1],
                                                                          label_length=label_positional_embedding.shape[1],
                                                                          label_padding_mask=token_padding_mask).to(context_with_mask.device)
        else:
            self_attn_padding_mask = None

        intermediate_output = []
        for i in range(self.num_layers):
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_attn_mask,
                tgt_key_padding_mask=self_attn_padding_mask,
                query_pos=query_pos,
            )

            output = self.transformer_ffn_layers[i](output)

            intermediate_output.append(self.decoder_norm(output).permute(1, 0, 2))

        output = self.decoder_norm(output) #N_patch, B, C

        output_dict = {'decoder_output': output.permute(1, 0, 2),
                       'decoder_output_reference': reference,
                       'decoder_output_aux': intermediate_output[:-1] if self.intermediate_output else None} # B, N_patch, C
        x.update(output_dict)

        return x

    def generate_self_attn_mask(self, patch_length, label_length, self_attn_mask_type='full', extra_text_num=0,
                                nested_padding_mask=None):
        # import pdb; pdb.set_trace()
        patch_label_length = patch_length + label_length
        max_length = patch_length + label_length + extra_text_num
        attention_mask = torch.ones((max_length, max_length), dtype=torch.bool)
        if self_attn_mask_type == 'full':
            attention_mask[:, :] = 0
        elif self_attn_mask_type == 'caption_mask':
            attention_mask = ~attention_mask
            label_causal_mask = ~(torch.triu(torch.ones(label_length, label_length)) == 1).transpose(0, 1)
            attention_mask[patch_length:, patch_length:] = label_causal_mask
            attention_mask[:patch_length, :] = True
            attention_mask[:patch_length, :patch_length][torch.eye(patch_length, dtype=torch.bool)] = False
        elif self_attn_mask_type == 'patch_diag_label_row':
            attention_mask[patch_length:patch_label_length, :patch_label_length] = 0
            attention_mask[:patch_length, :patch_length][torch.eye(patch_length, dtype=torch.bool)] = 0
        elif self_attn_mask_type == 'patch_diag_label_row_textlabelfull':
            attention_mask[patch_length:patch_label_length, :] = 0
            attention_mask[:patch_length, :patch_length][torch.eye(patch_length, dtype=torch.bool)] = 0
            attention_mask[patch_label_length:max_length, patch_length:] = 0
        elif self_attn_mask_type == 'patch_diag_label_row_nested':
            attention_mask[patch_length:patch_label_length, :patch_label_length] = 0
            attention_mask[:patch_length, :patch_length][torch.eye(patch_length, dtype=torch.bool)] = 0
            assert nested_padding_mask is not None
            batch_size = nested_padding_mask.shape[0]
            num_heads = self.num_heads
            attention_mask = attention_mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)
            for i in range(batch_size):
                attention_mask[i * num_heads:(i + 1) * num_heads,
                patch_length:patch_label_length,
                nested_padding_mask[i]] = 1
        elif self_attn_mask_type == 'full_nested':
            attention_mask[:, :] = 0
            assert nested_padding_mask is not None
            batch_size = nested_padding_mask.shape[0]
            num_heads = self.num_heads
            attention_mask = attention_mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)
            for i in range(batch_size):
                attention_mask[i * num_heads:(i + 1) * num_heads,
                patch_length:patch_label_length,
                nested_padding_mask[i]] = 1

        return attention_mask

    def generate_self_attn_padding_mask(self, patch_length, label_length, label_padding_mask=None):
        patch_label_length = patch_length + label_length
        padding_mask = torch.zeros(label_padding_mask.shape[0], patch_label_length)
        padding_mask[:, patch_length:] = label_padding_mask
        return padding_mask.bool()

    def generate_pos_with_pos_mode(self, batch_size, pos_embed, paramed_pe_shape, current_patch_shape, mode=False, padding_attn_masks=None):
        """
        False: directly add
        'simple_interpolate': interpolate positional embedding to current image size: detection
        :param batch_size: int. batch size
        :param pos_embed: Tensor, pos_embedding parameters
        :param paramed_pe_shape: int[h,w], the shape of the positional embedding parameters
        :param current_patch_shape: int[h,w] current input patch tokens shape
        :param mode: str/boolen.
        :return: generated positional embedding with shape of [batch_size, num_tokens, C]
        """
        if mode is False:
            positional_embedding = pos_embed.repeat(batch_size, 1, 1)
        elif mode == 'simple_interpolate':
            #  get the real input patch branch shape from adapter_output of the patch input

            pos_embed = pos_embed.reshape(-1, paramed_pe_shape[0], paramed_pe_shape[1],
                                          pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=current_patch_shape, mode='bicubic',
                                                        align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

            positional_embedding = pos_embed.repeat(batch_size, 1, 1)
        elif mode == 'interpolate_with_nomask':
            assert padding_attn_masks is not None
            batch_size = padding_attn_masks.shape[0]
            not_mask = ~padding_attn_masks
            h_list = not_mask.sum(1)[:,0]
            w_list = not_mask.sum(2)[:,0]

            pos_embed = pos_embed.reshape(-1, paramed_pe_shape[0], paramed_pe_shape[1],
                                          pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=current_patch_shape, mode='bicubic',
                                                        align_corners=False)
            pos_embed = pos_embed.repeat(batch_size, 1, 1, 1)

            for i in range(batch_size):
                pos_embed[i, :, :h_list[i], :w_list[i]] = torch.nn.functional.interpolate(
                    pos_embed[i, :, :, :].unsqueeze(0), size=(h_list[i], w_list[i]), mode='bicubic',
                    align_corners=False).squeeze(0)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            positional_embedding = pos_embed

        elif mode == 'smpl_xyz':
            #  in this mode, pos_embed are projected from anchor, instead of from self.query_embed_{label/patch}
            #  fix the number of vertices to 17*17
            #
            if (self.query_embed_label_z is None) or (self.query_embed_label_xy is None):
                self.prepare_smpl_PE()

            if self.smpl_mae_pe:
                positional_embeddings_xyz = self.adapt_pos2d(self.query_embed_label_xy) + self.adapt_pos1d(self.query_embed_label_z)

                camera_PE = torch.zeros(1,1,self.hidden_dim).float().cuda()

                # print("self.query_embed.shape {}".format(self.query_embed.shape))
                # print("self.query_embed_label.shape {}".format(self.query_embed_label.shape))
                positional_embedding = []
                positional_embedding.append(camera_PE.repeat(batch_size,1,1))
                positional_embedding.append(positional_embeddings_xyz.repeat(batch_size,1,1))


                positional_embedding = torch.cat(positional_embedding, dim=1) #  bs, 1 + 445, dim
            else:
                positional_embeddings_xyz_cam = self.adapt_pos2d(self.query_embed_label) 
                # print("self.query_embed.shape {}".format(self.query_embed.shape))
                # print("self.query_embed_label.shape {}".format(self.query_embed_label.shape))
                positional_embedding = []
                positional_embedding.append(positional_embeddings_xyz_cam.repeat(batch_size,1,1))


                positional_embedding = torch.cat(positional_embedding, dim=1) #  bs, 1 + 445, dim

        return positional_embedding

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def vertices_3dembed(vertices, num_vertices_feats=256, temperature=10000):
    """
    project vertices to 3d embedding
    :param vertices:
    :param num_vertices_feats:
    :param temperature:
    :return:
    """
    scale = 2 * math.pi
    vertices = vertices* scale
    dim_t = torch.arange(num_vertices_feats, dtype=torch.float32,)
    dim_t = temperature ** (2 * (dim_t // 2) / num_vertices_feats).cuda()
    # import pdb;pdb.set_trace()
    vertices_x = vertices[..., 0, None] / dim_t  # QxBx128
    vertices_y = vertices[..., 1, None] / dim_t
    vertices_z = vertices[..., 2, None] / dim_t
    # import pdb;pdb.set_trace()
    vertices_x = torch.stack((vertices_x[..., 0::2].sin(), vertices_x[..., 1::2].cos()), dim=-1).flatten(-2)
    vertices_y = torch.stack((vertices_y[..., 0::2].sin(), vertices_y[..., 1::2].cos()), dim=-1).flatten(-2)
    vertices_z = torch.stack((vertices_z[..., 0::2].sin(), vertices_z[..., 1::2].cos()), dim=-1).flatten(-2)
    # import pdb;pdb.set_trace()
    verticesemb = torch.cat((vertices_z, vertices_y, vertices_x), dim=-1)
    return verticesemb


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    # QxBx2
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t  # QxBx128
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)

    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed

