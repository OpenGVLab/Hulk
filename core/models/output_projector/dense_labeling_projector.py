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
import pdb
from collections import OrderedDict



class Norm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps
        self.normalized_shape = (embed_dim,)

        #  >>> workaround for compatability
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln.weight = self.weight
        self.ln.bias = self.bias

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def _get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DenseLabelingProjector(nn.Module):
    modality = 'dense_labeling'
    """
    Projector for dense labeling task, generating dense labeling.
    :param loss_cfg: loss config
    :param hidden_dim: hidden dimension
    :param patch_size: patch size of dense labeling tokens
    :param in_chans: input channels
    :param stride_level: stride level of input dense labeling
    :param task_sp_list: List of task specific list for DDP communication. Default: ()
    :param emb_padding_idx: padding index of embedding layer. Default: None
    :param text_prototype: whether to use text prototype from BERT-extracted features. Default: False
    :param text_dim: text dimension. Default: 768
    :param post_mul_norm: whether to use LayerNorm normalization to normalize the projected results after multiplication.
    :param ginfo: global info for DDP communication. Default: None
    :param cls_loss_branch: whether to use classification loss branch. Default: False. When True, the class embedding
                            is computed by mean pooling the reconstructed dense labeling features.
    :param text_embed_first_mul_second_inter: whether to first multiply the text embedding with the reconstructed dense
                            labeling features, and then use the intermediate result to predict the dense labeling maps.
    :param text_embed_project_with_MLP: whether to use MLP to project the text embedding to the same dimension as the
                            hidden dimension of the decoder. Default: False.
    """
    def __init__(self,
                 loss_cfg,
                 hidden_dim=256,
                 patch_size=[16, 16],
                 in_chans=20,
                 stride_level=4,
                 task_sp_list=(),
                 modality_share_list=(),
                 emb_padding_idx=None,
                 text_dim=768,
                 post_mul_norm=True,
                 post_mul_norm_cls=False,
                 ginfo=None,
                 cls_loss_branch=False,
                 text_embed_first_mul_second_inter=False,
                 text_embed_project_with_MLP=False,
                 replace_post_mul_norm=False,
                 translate_weight_scale=1.0,
                 description_dict_name="coco_pose",
                 task="parsing",
                 mask_times_cls=False,
                 upsample_before_product=False,
                 upsample_hidden_dim=256,
                 ):
        super().__init__()
        self.stride_level = stride_level
        self.P_H, self.P_W = max(1, patch_size[0] // stride_level), max(1, patch_size[1] // stride_level)
        self.num_channels = in_chans
        self.classes = in_chans

        self.text_dim = text_dim
        self.loss_cfg = loss_cfg
        self.cls_loss_branch = cls_loss_branch

        self.text_embed_first_mul_second_inter = text_embed_first_mul_second_inter
        self.text_embed_project_with_MLP = text_embed_project_with_MLP
        self.replace_post_mul_norm = replace_post_mul_norm
        self.translate_weight_scale = translate_weight_scale

        self.task = task
        self.mask_times_cls = mask_times_cls
        self.upsample_before_product = upsample_before_product
        self.up_hidden_dim = upsample_hidden_dim


        self.class_proj = nn.Linear(768, hidden_dim, bias=False)  # 768 is the dimension of BERT output
        self.patch_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.num_classes = in_chans

        loss_cfg.ginfo = ginfo

        # add the ignore idx both in the 'output_proj' and CE loss 'ignore_idx'
        if emb_padding_idx is not None:
            self.num_channels += 1
            # self.classes +=1
            # num of classes starts from 0, so the index of in_chans is the ignore_idx.
            loss_cfg['kwargs']['ignore_index'] = in_chans

        # load bert vectors
        text_features = torch.load(f'./{description_dict_name}.pth')
        self.text_features = nn.Parameter(torch.zeros(text_features.shape), requires_grad=False)
        self.text_features.data.copy_(text_features)

        if post_mul_norm:
            self.post_mul_norm = nn.LayerNorm(self.classes,eps=1e-6)
        else:
            assert self.replace_post_mul_norm
            self.post_mul_norm = None
            self.translate_weight = nn.Parameter(torch.ones([1]) * translate_weight_scale)
            self.translate_bias = nn.Parameter(torch.zeros([1]))

        if post_mul_norm_cls:
            self.post_mul_norm_cls = nn.LayerNorm(self.classes,eps=1e-6)
        else:
            self.post_mul_norm_cls = None

        if self.upsample_before_product:
            self.upsample_network = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, upsample_hidden_dim, kernel_size=2, stride=2),
                Norm2d(upsample_hidden_dim),
                _get_activation(activation='gelu'),
                nn.ConvTranspose2d(upsample_hidden_dim, hidden_dim, kernel_size=2, stride=2),
            )
        else:
            self.upsample_network = nn.Sequential(
                nn.ConvTranspose2d(self.classes, hidden_dim, kernel_size=2, stride=2),
                Norm2d(hidden_dim),
                _get_activation(activation='gelu'),
                nn.ConvTranspose2d(hidden_dim, self.classes, kernel_size=2, stride=2),
            )

        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list
        self.loss_fn = recons_loss_entry(loss_cfg)

    def forward_mask(self, dense_labeling_output, x):

        B, _, C = dense_labeling_output.shape

        # use learnable class embedding as queries in decoder
        patches = self.patch_proj(dense_labeling_output) # linear

        cls_seg_feat = self.class_proj(self.text_features)
        cls_seg_feat = cls_seg_feat.unsqueeze(0).repeat(B,1,1)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        if self.upsample_before_product:
            H, W = x.image.shape[-2:]
            # Number of patches in height and width
            N_H = H // (self.stride_level * self.P_H)
            N_W = W // (self.stride_level * self.P_W)
            patches = rearrange(patches, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)

            patches = self.upsample_network(patches)
            patches = rearrange(patches, "b c nh nw -> b (nh nw) c", nh=H // 4, nw=W // 4)
            patches = F.normalize(patches, dim=2, p=2)
            masks = patches @ cls_seg_feat.transpose(1, 2)
            masks = self.post_mul_norm(masks)
            masks = rearrange(masks, "b (nh nw) c -> b c nh nw", nh=H // 4, nw=W // 4)

        else:
            # first multiple (dync-conv) then interpolate
            patches = F.normalize(patches, dim=2, p=2)
            masks = patches @ cls_seg_feat.transpose(1, 2)
            masks = self.post_mul_norm(masks)

            H, W = x.image.shape[-2:]
            # Number of patches in height and width
            N_H = H // (self.stride_level * self.P_H)
            N_W = W // (self.stride_level * self.P_W)
            masks = rearrange(masks, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W)
            masks = self.upsample_network(masks)

        # Project to cls
        if self.cls_loss_branch:
            pred_classes = dense_labeling_output.mean(1, keepdim=True) # b, dimension

            if self.post_mul_norm_cls is not None:
                pred_classes = F.normalize(pred_classes, dim=2, p=2)
            pred_classes = (pred_classes @ cls_seg_feat.transpose(1, 2)).squeeze() # b, num_class
            if self.post_mul_norm_cls is not None:
                pred_classes = self.post_mul_norm_cls(pred_classes)
        else:
            pred_classes = None

        if self.mask_times_cls:
            pred_classes = pred_classes.sigmoid()
            masks = masks.sigmoid()

            semseg = torch.einsum("bq,bqhw->bqhw", pred_classes, masks)
            masks = semseg


        return masks, pred_classes


    def forward(self, x,  mask=None):
        """
        :param x: input dict, including 'backbone_masking_info', 'decoder_output', 'image' (augmented),
                        'label' (augmented), 'gt' (non-augmented), 'instance' (a class that contains the segmentation
                        binary masks), 'width' (non-augmented), 'height' (non-augmented).
        :param mask: None
        :return: loss_dict(train)/output_dict(eval). loss_dict contains 'losses'. output_dict contains 'preds'.
        """
        task_mask = x.backbone_masking_info.task_masks[self.modality]
        task_infos = x.backbone_masking_info.task_infos
        dense_labeling_output = x.decoder_output[:, task_infos['tasks']['dense_labeling']['start_idx']:task_infos['tasks']['dense_labeling']['end_idx']]

        B, _, C = dense_labeling_output.shape

        masks, pred_classes = self.forward_mask(dense_labeling_output, x)

        if self.training:
            if self.task == "parsing":
                outputs = {}
                outputs['pred_masks'] = masks
                if self.cls_loss_branch:
                    outputs['pred_classes'] = pred_classes
                targets = x.instances
                targets = prepare_targets_for_set_loss(targets, x.image, self.stride_level)

                if 'decoder_output_aux' in x and x.decoder_output_aux is not None:
                    aux_outputs = []
                    for aux_decoder_output in x.decoder_output_aux:
                        aux_dense_labeling_output = aux_decoder_output[:, task_infos['tasks']['dense_labeling']['start_idx']:task_infos['tasks']['dense_labeling']['end_idx']]
                        aux_dense_labeling_output_masks, aux_dense_labeling_output_pred_classes = self.forward_mask(aux_dense_labeling_output, x)
                        if aux_dense_labeling_output_pred_classes is None:
                            aux_outputs.append({'pred_masks': aux_dense_labeling_output_masks})
                        else:
                            aux_outputs.append({'pred_masks': aux_dense_labeling_output_masks, 'pred_classes': aux_dense_labeling_output_pred_classes})

                    outputs['aux_outputs'] = aux_outputs

                loss_dict = self.loss_fn(outputs, targets, task_mask, self.modality, )
                return loss_dict

            elif self.task == "pose":
                outputs = {}
                outputs['pred_masks'] = masks
                if pred_classes is not None:
                    outputs['pred_classes'] = pred_classes
                target = x
                target_weight = x['target_weight']

                if 'decoder_output_aux' in x and x.decoder_output_aux is not None:
                    aux_outputs = []
                    for aux_decoder_output in x.decoder_output_aux:
                        aux_dense_labeling_output = aux_decoder_output[:, task_infos['tasks']['dense_labeling']['start_idx']:task_infos['tasks']['dense_labeling']['end_idx']]
                        aux_dense_labeling_output_masks, aux_dense_labeling_output_pred_classes = self.forward_mask(aux_dense_labeling_output, x)
                        if aux_dense_labeling_output_pred_classes is None:
                            aux_outputs.append({'pred_masks': aux_dense_labeling_output_masks})
                        else:
                            aux_outputs.append({'pred_masks': aux_dense_labeling_output_masks, 'pred_classes': aux_dense_labeling_output_pred_classes})

                    outputs['aux_outputs'] = aux_outputs

                loss_dict = self.loss_fn(outputs, target, target_weight)

                return loss_dict
            else:
                raise ValueError("task ({}) is not supported!".format(self.task))
        else:
            if self.task == "parsing":
                output = []
                dense_labeling_output = F.interpolate(masks,
                                                      size=(x.image.shape[-2], x.image.shape[-1]), mode="bilinear",
                                                      align_corners=False, )
                for _idx, dense_labeling_out in enumerate(dense_labeling_output):
                    try:
                        height = x.get("gt", None).shape[-2]  # .item()
                        width = x.get("gt", None).shape[-1]  # .item()
                    except:
                        height = x['height'][_idx].item()
                        width = x['width'][_idx].item()
                    image_size = (x.image.shape[-2], x.image.shape[-1])

                    output.append({})

                    r = sem_seg_postprocess(dense_labeling_out, image_size, height, width)

                    output[-1]["sem_seg"] = r
                return output
                    # pdb.set_trace()
            elif self.task == "pose":
                output = []
                assert x['image'].size(0) == len(x['img_metas'])
                batch_size, _, img_height, img_width = x['image'].shape
                if batch_size > 1:
                    assert 'bbox_id' in x['img_metas'][0].data
                output_heatmap = pose_postprocess(masks, flip_pairs=None)
                preds = {'output_heatmap': output_heatmap, 'pred_logits': pred_classes}
                return preds


def prepare_targets_for_set_loss(targets, images, stride_level):  # for seg
    """
    :param targets: target list. each element is a dict with keys: "gt_classes", "gt_masks"
    :param images: a batch of images
    :param stride_level: the stride level of the input images. e.g. 4 for MultiMAE, we use 1 as default
    :return: new_targets: a list of dict with keys: "labels", "masks".
    """
    h_pad, w_pad = images.shape[-2:]
    h_pad, w_pad = h_pad // stride_level, w_pad // stride_level
    new_targets = []
    for targets_per_image in targets:
        # pad gt TODO: seems duplicated?
        gt_masks = targets_per_image.gt_masks
        padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
        new_targets.append(
            {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            }
        )
    return new_targets

def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result

def pose_postprocess(x, flip_pairs=None):
    """Inference function.

    Returns:
        output_heatmap (np.ndarray): Output heatmaps.

    Args:
        x (torch.Tensor[NxKxHxW]): Input features.
        flip_pairs (None | list[tuple()):
            Pairs of keypoints which are mirrored.
    """
    if flip_pairs is not None:
        output_heatmap = flip_back(
            x.detach().cpu().numpy(),
            flip_pairs,
            target_type=self.loss.target_type)
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if self.test_cfg.get('shift_heatmap', False):  # True
            output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
    else:
        output_heatmap = x.detach().cpu().numpy()
    return output_heatmap

def pose_decode(self, img_metas, output, **kwargs):
    """Decode keypoints from heatmaps.

    Args:
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
    """
    batch_size = len(img_metas)

    if 'bbox_id' in img_metas[0].data:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    image_paths = []
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i].data['center']
        s[i, :] = img_metas[i].data['scale']
        image_paths.append(img_metas[i].data['image_file'])

        if 'bbox_score' in img_metas[i].data:
            score[i] = np.array(img_metas[i].data['bbox_score']).reshape(-1)
        if bbox_ids is not None:
            bbox_ids.append(img_metas[i].data['bbox_id'])

    preds, maxvals = keypoints_from_heatmaps(
        output,
        c,
        s,
        unbiased=self.test_cfg.get('unbiased_decoding', False),
        post_process=self.test_cfg.get('post_process', 'default'),
        kernel=self.test_cfg.get('modulate_kernel', 11),
        valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                              0.0546875),
        use_udp=self.test_cfg.get('use_udp', False),
        target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

def dense_labeling_projector(**kwargs):
    """
    Create a projector for dense labeling task using the given config.
    :param kwargs:
    :return: projector
    """
    default = dict()
    recursive_update(default, kwargs)
    # import pdb; pdb.set_trace()
    projector = DenseLabelingProjector(**default)

    return projector
