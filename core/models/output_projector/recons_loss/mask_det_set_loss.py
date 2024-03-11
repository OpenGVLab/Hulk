import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import DetectionHungarianMatcher
from .mask_set_loss import MaskDetSetCriterion

class MaskDetFocalDiceLoss(nn.Module):
    def __init__(self, cfg,
                 patch_size=[None,None],  # placeholder, achieve similar kwargs with other adapters
                 stride=None,
                 ginfo=None,
                 ):
        super(MaskDetFocalDiceLoss, self).__init__()
        matcher = DetectionHungarianMatcher(
            cost_class=cfg.class_weight,
            cost_bbox=cfg.bbox_weight,
            cost_giou=cfg.giou_weight,
        )

        modality = 'sparse_labeling'
        weight_dict = {f"loss_{modality}_ce": cfg.class_weight,
                       f"loss_{modality}_bbox": cfg.bbox_weight,
                       f"loss_{modality}_giou": cfg.giou_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers-1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.fd_loss = MaskDetSetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["labels", "boxes"],
            focal_alpha=cfg.focal_alpha,
            ign_thr=cfg.ign_thr,
            predict3d=cfg.get('predict3d',False),
            xyxy=cfg.get('xyxy',False),
        )

        self.cfg = cfg

    def forward(self, outputs, raw_targets, mask=None, modality='', **kwargs): # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, raw_targets, mask=mask, modality=modality)
        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            elif 'loss' in k:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses