import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.ops.utils import ShapeSpec
from core.models.ops.utils import c2_xavier_fill

from core.utils import NestedTensor


def _get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class MAEdecoder_proj_neck(nn.Module):
    """
    the feature projection neck for MAE decoder with only a Linear layer, a LayerNorm (optional) and a type embedding
    (optional). The type embedding is used for the decoder with a smaller dimension than that in the encoder.
    :param mask_dim: int. the dimension of the mask
    :param backbone: placeholder
    :param task_sp_list: list. task specific list for DDP communication. Default: ().
    :param mask_forward: bool. whether to forward the mask. Default: True.
    :param modality: str. the modality of the input. Default: 'rgb'.
    :param type_embed: bool. whether to use type embedding. Default: False.
    :param type_embed_zero_init: bool. whether to initialize the type embedding with zeros. Always lead to better
        performance when True .Default: False.
    :param neck_layernorm: bool. whether to use LayerNorm in the neck. Default: False.
    """
    def __init__(self,
                 mask_dim,
                 backbone,  # placeholder
                 task_sp_list=(),
                 mask_forward=True,
                 modality='rgb',
                 type_embed=False,
                 type_embed_zero_init=False,
                 neck_layernorm=False,
                 conv_neck=False,
                ):
        super(MAEdecoder_proj_neck, self).__init__()
        self.task_sp_list = task_sp_list
        self.modality = modality

        self.vis_token_dim = self.embed_dim = backbone.embed_dim
        self.mask_dim = mask_dim
        self.neck_layernorm = neck_layernorm
        self.conv_neck = conv_neck

        self.mask_map = nn.Sequential(
            nn.Linear(self.embed_dim, mask_dim, bias=True)
        ) if mask_dim else False
        if self.conv_neck:
            self.mask_map = nn.Sequential(
                nn.Conv2d(self.embed_dim, mask_dim, 1)
            )
        self.neck_ln = nn.LayerNorm(mask_dim) if neck_layernorm else False

        self.mask_forward = mask_forward
        #  the type embedding is used for the decoder with a smaller dimension than that in the encoder
        self.task_embed_decoder = nn.Embedding(1, mask_dim) if type_embed else None
        if type_embed and type_embed_zero_init:
            self.task_embed_decoder.weight.data = torch.zeros(1, mask_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # to be compatible with unihcpv1, neck still using mask_map as the projector,
        # and mask_features as the neck_output, will be deprecated in the future
        # import pdb;pdb.set_trace()
        if self.neck_ln and not self.conv_neck:
            if self.mask_map and self.mask_forward:
                features.update({f'neck_output_{self.modality}': {'mask_features': self.neck_ln(self.mask_map(features['backbone_output'])),
                                                'multi_scale_features': [features['backbone_output']],
                                                                f'task_embed_decoder': self.task_embed_decoder,
                                                                }})
            else:
                features.update({'neck_output': {'mask_features': None,
                                                'multi_scale_features': [features['backbone_output']]}})
        elif self.conv_neck:
            #  only for v2 det detection
            Hp = features.adapter_output_rgb.N_H
            Wp = features.adapter_output_rgb.N_W
            B = features.backbone_output.shape[0]

            proj_feats = self.mask_map(features['backbone_output'].permute(0, 2, 1).reshape(B, -1, Hp, Wp)).flatten(2, 3).permute(0, 2, 1)

            if self.neck_ln:
                proj_feats = self.neck_ln(proj_feats)
            if self.mask_map and self.mask_forward:
                features.update({f'neck_output_{self.modality}': {'mask_features': proj_feats,
                                                'multi_scale_features': [features['backbone_output']],
                                                                f'task_embed_decoder': self.task_embed_decoder,
                                                                }})
        else:
            if self.mask_map and self.mask_forward:
                features.update({f'neck_output_{self.modality}': {'mask_features': self.mask_map(features['backbone_output']),
                                                'multi_scale_features': [features['backbone_output']],
                                                                f'task_embed_decoder': self.task_embed_decoder,
                                                                }})
            else:
                features.update({'neck_output': {'mask_features': None,
                                                'multi_scale_features': [features['backbone_output']]}})                          

        return features



