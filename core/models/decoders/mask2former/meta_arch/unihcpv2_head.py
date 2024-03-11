# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.memory import retry_if_cuda_oom
from core.data.transforms.post_transforms import pose_pck_accuracy, flip_back, transform_preds
from core.models.ops import box_ops
from ..transformer_decoder import Hulk_Decoder
from ast import literal_eval


class UniHCPv2_Head(nn.Module):
    def __init__(self,
                 transformer_predictor_cfg,
                 loss_cfg, # placeholder
                 num_classes,
                 backbone,  # placeholder
                 neck,  # placeholder
                 patch_adapter,
                 label_adapter,
                 patch_neck,
                 label_neck,
                 loss_weight,
                 ignore_value,
                 ginfo,
                 bn_group,  # placeholder
                 task_sp_list=(),
                 neck_sp_list=(),
                 modality_share_list=(),
                 task='recons_rgb',
                 test_cfg=None,
                 predictor='hulk',
                 interpolate_mae_pe=True,
                 ):

        """
        A unified head for all tasks in UniHCPv2. The only support task is "recons_rgb".
        :param transformer_predictor_cfg: dict, config for transformer decoder
        :param loss_cfg: placeholder. the loss config is specified in the config file of each projector
        :param num_classes: int, number of classes to predict
        :param backbone: placeholder
        :param neck: placeholder
        :param patch_adapter: nn.Module, patch adapter
        :param label_adapter: nn.Module, label adapter
        :param patch_neck: nn.Module, patch neck, to provide task token for patch tokens in transformer decoder
        :param label_neck: nn.Module, label neck, to provide task token for label tokens in transformer decoder
        :param loss_weight: placeholder
        :param ignore_value: placeholder
        :param ginfo: global info for transformer decoder
        :param bn_group: placeholder
        :param task_sp_list: specify params/buffers in decoder that should be treated task-specific in reduce_gradients()
        :param neck_sp_list: specify params/buffers in decoder that should be treated neck-specific in reduce_gradients()
        :param task: str, the task to perform. Only support "recons_rgb" now.
        :param test_cfg: placeholder. The test config is specified in the config file of each projector
        :param predictor: str, the predictor to use. Support "mae" and "mae_withtext" now.
        :param feature_only: bool, redundant param in compliance with past reid test code
        :param interpolate_mae_pe: bool, whether to interpolate the positional embedding in MAE
        """
        super().__init__()
        self.task = task
        self.task_sp_list = task_sp_list
        self.neck_sp_list = neck_sp_list
        self.modality_share_list = modality_share_list

        self.backbone = [backbone]  # avoid recursive specific param register
        self.neck = [neck]  # avoid recursive specific param register
        self.patch_adapter = [patch_adapter] # avoid recursive specific param register
        self.label_adapter = [label_adapter]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        if predictor == 'hulk':
            self.predictor = Hulk_Decoder(in_channels=neck.vis_token_dim,
                                          mask_dim=neck.mask_dim,
                                          patch_size=patch_adapter.patch_size,
                                          num_patches=patch_adapter.num_patches,
                                          patch_shape=patch_adapter.patch_shape,
                                          patch_shape_label=label_adapter.patch_shape,
                                          interpolate_mae_pe=interpolate_mae_pe,
                                          mask_classification=True,
                                          num_classes=num_classes,
                                          ginfo=ginfo,
                                          backbone_pose_embed=patch_adapter.pos_embed, # dumped, deleted in the future
                                          **transformer_predictor_cfg)

        #   set the patch and label task token using the corresponding type embedding in the neck
        self.predictor.patch_task_token = patch_neck.task_embed_decoder
        self.predictor.label_task_token = label_neck.task_embed_decoder

    def forward(self, features):  # input -> loss, top1, etc.
        # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx, 'neck_output': xxx}
        if self.task == 'recons':
            outputs = self.predictor.forward_recons(features)
            return outputs
        else:
            raise NotImplementedError
