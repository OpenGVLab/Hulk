# Base on M-MAE
# https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/input_adapters.py
import os
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

from .rgb_adapter import RGBAdapter

class DenseLabelingAdapter(RGBAdapter):
    """
    Dense labeling adapter for dense labeling task, i.e., semantic segmentation.
    for a specific adapter, it should include "Normalizer, Projector, Position Embed".
    :param: dim_class_embed: the dimension of class embedding. Class embedding is used to embed the class label.
    :param: img_size: the size of input image.
    :param: patch_size: the size of patch.
    :param: in_chans: the number of input channels, which denotes the number of classes.
    :param: embed_dim: the dimension of embedding.
    :param: stride_level: the stride level of patch. The stride level is used to control the size of patch.
    The size of patch is img_size/stride_level. When stride_level=4, the input dense label map is downsampled by 4.
    :param: pad_attn_mask: whether to pad the attention mask.
    :param: test_pos_mode: whether to use test position mode.
    :param: use_abs_pos_emb: whether to use absolute position embedding. Default: False
    :param: learnable_pos: whether to use learnable position embedding. A corresponding parameter of use_abs_pos_emb. Default: False
    :param: round_padding: whether to use round padding. Default: False
    :param: task_sp_list: List of task specific list for DDP communication. Default: ()
    :param: type_embed: whether to use type embedding. Default: True
    :param: emb_padding_idx: the padding index of embedding. emb_padding_idx is used to pad the class embedding.
    :param: type_embed_zero_init: whether to use zero initialization for type embedding. Default: False
    """
    def __init__(self,
                 dim_class_embed=64,
                 img_size=224,
                 patch_size=16,
                 in_chans=20,
                 embed_dim=768,
                 stride_level=1,
                 pad_attn_mask=False,
                 test_pos_mode=False,
                 use_abs_pos_emb=False,
                 learnable_pos=False,
                 round_padding=False,
                 task_sp_list=(),
                 modality_share_list=(),
                 type_embed=True,
                 emb_padding_idx=None,
                 type_embed_zero_init=False,
                 ):

        super(DenseLabelingAdapter, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                            embed_dim=embed_dim, stride_level=stride_level, pad_attn_mask=pad_attn_mask,
                                            test_pos_mode=test_pos_mode, use_abs_pos_emb=use_abs_pos_emb,
                                            learnable_pos=learnable_pos, round_padding=round_padding,
                                            modality_share_list=modality_share_list,
                                            task_sp_list=task_sp_list, type_embed=type_embed,type_embed_zero_init=type_embed_zero_init)
        self.num_classes = in_chans
        self.dim_class_embed = dim_class_embed
        self.emb_padding_idx = emb_padding_idx

        if self.emb_padding_idx is not None:
            self.emb_padding_idx = self.num_classes
            self.num_classes += 1

        self.class_embed = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.dim_class_embed,
                                        padding_idx=self.emb_padding_idx)
        trunc_normal_(self.class_embed.weight, std=0.02)

        self.proj = nn.Conv2d(in_channels=self.dim_class_embed,
                          out_channels=embed_dim,
                          kernel_size=(self.P_H, self.P_W),
                          stride=(self.P_H, self.P_W))

        self.initialize_proj_weights()

    @staticmethod
    def _normalization(x):
        """normalize the input dense label map."""
        return x

    def forward(self, input_var):
        """
        :param input_var: (B, H, W), each element is the bert feature of the class name of each pixel.
        :return: input_var: adapter_output_dense_labeling: {'tokens': x_patch,  # [B, N_H*N_W, embed_dim]
                                                            'N_H': num_patch_H,
                                                            'N_W': num_patch_W,
                                                            'Bs': Batch_size,
                                                            'attn_mask': attn_mask}
        """
        output = {}

        if not self.training:
            B, _, H, W = input_var.image.shape
            H, W = H // self.stride_level, W // self.stride_level
            N_H, N_W = H // self.P_H, W // self.P_W
            x_patch = torch.zeros([B, N_H*N_W, self.embed_dim]).to(input_var.image.device)
            output['adapter_output_dense_labeling'] = {'tokens': x_patch,
                                                       'N_H': N_H,
                                                       'N_W': N_W,
                                                       'Bs': B,
                                                       'attn_mask': None}
            input_var.update(output)

            return input_var

        if 'dense_labeling' in input_var:
            x = input_var['dense_labeling']
        else:
            x = input_var['label']

        if x.dim() == 4: # for pose estimation
            B, _, H, W  = x.shape
        elif x.dim() == 3: # for human parsing
            B, H, W = x.shape

        N_H, N_W = H // self.P_H, W // self.P_W

        x = self._normalization(x)

        # Map to embedding
        if x.dim() == 3:
            # for human parsing
            x = rearrange(self.class_embed(x), 'b nh nw c -> b c nh nw')
        elif x.dim() == 4:
            x = torch.argmax(x, dim=1)
            x = rearrange(self.class_embed(x), 'b nh nw c -> b c nh nw')

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')
        x_patch = self.forward_PE(x_patch, N_H, N_W)

        if self.type_embed is not None:
            x_patch = self.with_type_embed(x_patch)

        output['adapter_output_dense_labeling'] = {'tokens': x_patch,
                                        'N_H': N_H,
                                        'N_W': N_W,
                                        'Bs': B,
                                        'attn_mask':None}
        input_var.update(output)

        return input_var


def dense_labeling_adapter(pretrained=False, **kwargs):
    """
    Create a dense labeling adapter.
    :param pretrained: no checkpoint is available for dense labeling adapter.
    :param kwargs:
    :return: DenseLabelingAdapter
    """
    default = dict(
        img_size=224, use_abs_pos_emb=True,   # as in table 11
        ####
    )
    recursive_update(default, kwargs)
    adapter = DenseLabelingAdapter(**default)

    return adapter
