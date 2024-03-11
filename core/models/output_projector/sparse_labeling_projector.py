import pdb
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dict_recursive_update import recursive_update
import numpy as np
from .recons_loss import recons_loss_entry
import os
from core.models.ops import box_ops

class SpatialTemporalSeq2DProjector(nn.Module):
    modality = 'sparse_labeling'
    """
    Projector for 2d spatial temporal sequence input, like keypoints sequences.
    Reconstruct the keypoints sequence from the spatial temporal cubes.
    :param loss_cfg: dict, config for the loss
    :param in_chans: int, number of input channels
    :param num_joints: int, number of joints
    :param num_frames: int, number of frames
    :param hidden_dim: int, hidden dimension of the decoder
    :param patch_size: tuple, temporal and spatial patch size. Eg. patch_size=(14, 5)
    :param stride_level: (tuple): temporal and spatial stride number. Eg. stride=(1,2)
            for real stride (patch_size[0], patch_size[1]//2)
    :param task_sp_list: list, task specific list for DDP communication. Default: ()
    :param proj_norm: str, normalization for the projector. Default: 'none'

    """

    def __init__(self,
                 loss_cfg: Dict,
                 in_chans: int,
                 num_joints: int,
                 num_frames: int,
                 hidden_dim: int,
                 patch_size: Tuple[int, int],
                 stride_level: Tuple[int, int],
                 task_sp_list: List = (),
                 modality_share_list: List = (),
                 proj_norm: str = 'none',
                 pre_proj_type: str = '',
                 num_classes: int = 1,
                 reference_type: str = '+-xyxy',
                 box_mlp: bool = False,
                 text_prototype: bool = False,
                 text_dim: int = 768,
                 learn_text: bool = False,
                 replace_post_mul_norm: bool = False,
                 translate_weight_scale: float = 1.0,
                 contrast_on_last_layer_text_tokens: bool = False,
                 description_dict_name='pedestrian_detection_name',
                 task: str = 'detection',
                 pred_joints_class: bool = False,
                 ):
        super().__init__()
        self.in_channels = in_chans
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.stride_level = stride_level
        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list
        
        self.proj_norm = proj_norm
        self.loss_cfg = loss_cfg
        self.patch_size = patch_size
        self.pre_proj_type = pre_proj_type
        self.num_classes = num_classes
        self.post_mul_norm = nn.LayerNorm(num_classes)
        self.reference_type = reference_type
        self.box_mlp = box_mlp
        self.text_prototype = text_prototype
        self.text_dim = text_dim
        self.learn_text = learn_text
        self.replace_post_mul_norm = replace_post_mul_norm
        self.contrast_on_last_layer_text_tokens = contrast_on_last_layer_text_tokens
        self.task = task
        self.pred_joints_class = pred_joints_class


        #  real stride when patching
        self.P_T = max(1, self.patch_size[0] // self.stride_level[0])
        self.P_J = max(1, self.patch_size[1] // self.stride_level[1])

        self.patch_shape = ((num_frames - self.patch_size[0]) // self.P_T + 1,
                            (num_joints- self.patch_size[1]) // self.P_J +1 )  # could be dynamic
        self.num_patches = ((num_frames - self.patch_size[0]) // self.P_T + 1) * \
                           ((num_joints- self.patch_size[1]) // self.P_J +1) # could be dynamic

        #  the output shape of the patching is [num_patches, B*M(M=2), hidden_dim],
        #  the reconstruct target is [B, M, num_joints, num_frames, in_channels]

        self.output_proj = nn.Linear(hidden_dim, self.P_T * self.P_J * self.in_channels, bias=True)
        if self.box_mlp:
            self.output_proj = MLP(hidden_dim, hidden_dim, self.P_T * self.P_J * self.in_channels, 3)

        self.class_proj = nn.Linear(self.text_dim, hidden_dim, bias=False)
        self.patch_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.translate_weight = nn.Parameter(torch.ones([1]) * translate_weight_scale)
        self.translate_bias = nn.Parameter(torch.zeros([1]))

        self.text_features = nn.Parameter(torch.randn(num_classes, self.text_dim), requires_grad=False)
        self.text_features.data.copy_(torch.load(f'./{description_dict_name}.pth'))

        self.loss_cfg.kwargs.cfg.ginfo = self.loss_cfg.kwargs.pop('ginfo')   # ginfo is added in the solver_mae_devnew.py create model
        self.loss_fn = recons_loss_entry(self.loss_cfg)

    def forward_points(self, sparse_labeling_output, reference=None):
        sparse_labeling_output_points = self.output_proj(sparse_labeling_output)  # points

        if reference is not None:
            if self.reference_type == 'four_points':
                sparse_labeling_output_points[..., :self.in_channels] += reference.transpose(0, 1)
                sparse_labeling_output_points[..., self.in_channels:] += reference.transpose(0, 1)
                sparse_labeling_output_points = sparse_labeling_output_points.sigmoid()
                sorted_x = torch.sort(sparse_labeling_output_points[..., 0::self.in_channels], dim=-1)[0]
                sorted_y = torch.sort(sparse_labeling_output_points[..., 1::self.in_channels], dim=-1)[0]
                if self.in_channels==3:
                    sorted_z = torch.sort(sparse_labeling_output_points[..., 2::self.in_channels], dim=-1)[0]
                    sparse_labeling_output_points_new = torch.cat([sorted_x[..., 0].unsqueeze(-1), sorted_y[..., 0].unsqueeze(-1), sorted_z[..., 0].unsqueeze(-1),
                                                                   sorted_x[..., 1].unsqueeze(-1), sorted_y[..., 1].unsqueeze(-1), sorted_z[..., 1].unsqueeze(-1),], dim=-1)
                else:
                    sparse_labeling_output_points_new = torch.cat([sorted_x[..., 0].unsqueeze(-1), sorted_y[..., 0].unsqueeze(-1),
                                                                   sorted_x[..., 1].unsqueeze(-1), sorted_y[..., 1].unsqueeze(-1),], dim=-1)
            elif self.reference_type == 'residual':
                sparse_labeling_output_points[..., :2] += reference.transpose(0, 1)
                sparse_labeling_output_points_new = sparse_labeling_output_points.sigmoid()
        else:
            sparse_labeling_output_points_new = sparse_labeling_output_points
        return sparse_labeling_output_points_new

    def forward_class(self, sparse_labeling_output, text_tokens=None):

        assert text_tokens is not None
        cls_text_feat = text_tokens
        cls_text_feat = self.class_proj(cls_text_feat)
        cls_text_feat = F.normalize(cls_text_feat, dim=2, p=2)

        patches = sparse_labeling_output
        patches = self.patch_proj(patches)
        patches = F.normalize(patches, dim=2, p=2)

        sim_matrix = torch.einsum('bnd, bcd->bnc', patches, cls_text_feat)  # class prediction,

        if self.replace_post_mul_norm:
            sim_matrix = sim_matrix*self.translate_weight + self.translate_bias
        else:
            sim_matrix = self.post_mul_norm(sim_matrix)

        return sim_matrix # [bs, num_patches, num_class]

    def forward(self, x, mask=None,):
        task_mask = x.backbone_masking_info.task_masks[self.modality]
        task_infos = x.backbone_masking_info.task_infos
        sparse_labeling_output = x.decoder_output[:, task_infos['tasks']['sparse_labeling']['start_idx']:task_infos['tasks']['sparse_labeling']['end_idx']]

        #  predict points
        if task_mask is not None:
            if task_mask.sum() != len(task_mask.flatten()): # use random-mask as aug only, not for loss
                loss = torch.tensor(0).to(task_mask.device)
                return {f'loss_{self.modality}_fake': loss,}
        sparse_labeling_output_points = self.forward_points(sparse_labeling_output, x.decoder_output_reference)

        #  predict classes
        if self.task == 'smpl':
            # Shape of patches in temporal and joint
            N_T, N_J = self.patch_shape
            # Reshape sequence of patches into image
            sparse_labeling_output_points = rearrange(sparse_labeling_output_points,
                                                      'b (nt nj) (c pt pj) -> b c (nt pt) (nj pj)', nt=N_T, nj=N_J,
                                                      pt=self.P_T, pj=self.P_J, c=self.in_channels)

        text_tokens = self.text_features.unsqueeze(0).repeat(sparse_labeling_output.shape[0], 1, 1)

        sim_matrix = self.forward_class(sparse_labeling_output, text_tokens)

        outputs = {'pred_points': sparse_labeling_output_points, 'pred_logits': sim_matrix}

        if 'decoder_output_aux' in x and x.decoder_output_aux is not None:
            aux_outputs = []  # list of dict [{'pred_points': sparse_labeling_output_points, 'pred_class': sim_matrix}, ...}]
            for aux_decoder_output in x.decoder_output_aux:
                aux_sparse_labeling_output = aux_decoder_output[:, task_infos['tasks']['sparse_labeling']['start_idx']:task_infos['tasks']['sparse_labeling']['end_idx']]
                aux_sparse_labeling_output_points = self.forward_points(aux_sparse_labeling_output, x.decoder_output_reference)
                if self.pre_proj_type == 'with_decoded_text_tokens':
                    if self.contrast_on_last_layer_text_tokens:
                        aux_text_tokens = x.decoder_output[:, -self.num_classes:]
                    else:
                        aux_text_tokens = aux_decoder_output[:, -self.num_classes:]
                elif self.pre_proj_type == 'fix_text_tokens':
                    aux_text_tokens = self.text_features.unsqueeze(0).repeat(aux_sparse_labeling_output.shape[0], 1, 1)
                else:
                    aux_text_tokens = None
                aux_sim_matrix = self.forward_class(aux_sparse_labeling_output, aux_text_tokens)
                aux_outputs.append({'pred_points': aux_sparse_labeling_output_points, 'pred_logits': aux_sim_matrix})
            outputs['aux_outputs'] = aux_outputs

        if self.training:
            loss_dict = self.loss_fn(outputs, raw_targets=x, mask=task_mask, modality=self.modality)
            return loss_dict
        else:
            if self.reference_type in ['four_points',]:
                results = ped_det_postprocess(outputs, x.orig_size, xyxy=True, point_dim=self.in_channels)
            elif self.reference_type == 'residual':
                results = ped_det_postprocess(outputs, x.orig_size, xyxy=False, point_dim=self.in_channels)
            elif self.reference_type == 'smpl':
                loss_dict = self.loss_fn(outputs, raw_targets=x, mask=task_mask, modality=self.modality)
                results = {'sparse_label_pred': sparse_labeling_output_points}
                results.update(loss_dict)
            else:
                results = {'sparse_label_pred': sparse_labeling_output_points}

            return results

    def prepare_detection_targets(self, raw_targets, xyxy=False):
        new_targets = []
        for targets_per_image in raw_targets:
            valid_len = (targets_per_image.area>0).sum()
            iscrowd = targets_per_image.iscrowd[:valid_len]
            boxes = targets_per_image.boxes[:valid_len]
            if xyxy:
                #   as boxes are point sequence, targets are format in xyxy.
                boxes = box_ops.box_xyxy_to_cxcywh(boxes)
            labels = targets_per_image.labels[:valid_len]

            new_targets.append(
                {
                    "boxes": boxes[~iscrowd],
                    "labels": labels[~iscrowd],
                    "area": targets_per_image.area[:valid_len],
                    "iscrowd": targets_per_image.iscrowd[:valid_len],
                    'ignore': boxes[iscrowd]
                }
            )

        return new_targets



def ped_det_postprocess(outputs, target_sizes, xyxy=True, point_dim=2,):
    """ Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits = outputs['pred_logits']

    pred_points = outputs.pop('pred_points')
    outputs['pred_boxes'] = torch.cat([pred_points[:, :, :2], pred_points[:, :, point_dim:point_dim+2]], dim=-1)
    if xyxy:
        #  sparse labeling predict boxes is xyxy
        out_bbox = box_ops.box_xyxy_to_cxcywh(outputs['pred_boxes'])
    else:
        out_bbox = outputs['pred_boxes']
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()

    # find the topk predictions
    num = out_logits.view(out_logits.shape[0], -1).shape[1]
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results

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

#  change the name from spatialtemporalseq2d_projector to sparse_labeling_projector
def sparse_labeling_projector(**kwargs):
    """
    Create a sparselabeling projector.
    :param kwargs:
    :return: projector
    """
    default = dict()
    recursive_update(default, kwargs)
    projector = SpatialTemporalSeq2DProjector(**default)

    return projector


