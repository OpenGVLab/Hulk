from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dict_recursive_update import recursive_update
import numpy as np
from .recons_loss import recons_loss_entry

class RGBProjector(nn.Module):
    modality = 'rgb'
    """
    Projecting the decoder output to RGB image
    :param: loss_cfg: the config for the loss function
    :param: hidden_dim: the hidden dimension of the decoder output
    :param: patch_size: the patch size of the output image
    :param: in_chans: the number of channels of the output image
    :param: stride_level: the stride level of the output image
    :param: task_sp_list: List of task specific list for DDP communication. Default: ()
    :param: ginfo: the global info for the DDP communication. Default: None
    :param: distribution_loss_on_adapter_tokens: whether to apply the distribution loss on the adapter tokens. Default: False
    :param: distribution_loss_cfg: the config for the distribution loss. Default: {}
            ### Note that this config is set here for quick implementation,                      ###
            ### the loss is implemented on adapter output tokens instead of decoded features.    ###
    """
    def __init__(self,
                 loss_cfg,
                 hidden_dim = 256,
                 patch_size = [16, 16],
                 in_chans = 3,
                 stride_level = 1,
                 task_sp_list = (),
                 modality_share_list = (),
                 ginfo = None,
                 ):
        super().__init__()
        self.stride_level = stride_level
        self.P_H, self.P_W = max(1, patch_size[0] // stride_level), max(1, patch_size[1] // stride_level)
        self.num_channels = in_chans

        self.output_proj = nn.Linear(hidden_dim, self.P_H * self.P_W * self.num_channels,
                                     bias=True)

        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list

        loss_cfg.kwargs.ginfo = ginfo
        self.loss_fn = recons_loss_entry(loss_cfg)


    def forward(self, x, mask=None):
        """
        :param x: input dict, including 'backbone_masking_info', 'decoder_output', 'image' (augmented),
                        'label' (augmented), 'gt' (non-augmented), 'instance' (a class that contains the segmentation
                        binary masks), 'width' (non-augmented), 'height' (non-augmented).
        :param mask:
        :return: loss_dict(train)/output(eval). loss dict contains the losses. output contains the output image.
        """
        task_mask = x.backbone_masking_info.task_masks[self.modality]
        task_infos = x.backbone_masking_info.task_infos
        rgb_output = x.decoder_output[:,task_infos['tasks']['rgb']['start_idx']:task_infos['tasks']['rgb']['end_idx']]

        B, _, C = rgb_output.shape

        H, W = x.image.shape[-2:]
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Project each token to (C * P_H * P_W)
        rgb_output = self.output_proj(rgb_output)

        # Reshape sequence of patches into image
        rgb_output = rearrange(rgb_output, 'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', nh=N_H, nw=N_W, ph=self.P_H, pw=self.P_W, c=self.num_channels)

        loss_dict = self.loss_fn(rgb_output, x.image, mask=task_mask, modality=self.modality)

        if self.training:
            return loss_dict
        else:
            output = {'rgb_pred': rgb_output}
            return output

def rgb_projector(**kwargs):
    """
    Create a RGBProjector instance.
    :param kwargs:
    :return: projector
    """
    default = dict()
    recursive_update(default, kwargs)
    projector = RGBProjector(**default)

    return projector