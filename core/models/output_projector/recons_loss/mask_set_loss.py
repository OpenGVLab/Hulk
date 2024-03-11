import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange
from core.utils import nested_tensor_from_tensor_list
from core.models.ops.utils import cat

from core.models.ops import box_ops

from core.data.transforms.post_transforms import pose_pck_accuracy, flip_back, transform_preds

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def allreduce(x, group_idx=None, ):
    if group_idx == 0:
        group_idx = None
    return dist.all_reduce(x,  group=group_idx)

def ratio2weight(targets, ratio):
    # import pdb;
    # pdb.set_trace()
    ratio = torch.from_numpy(ratio).type_as(targets)

    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def dice_loss_with_mask(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        loss_mask: torch.Tensor
    ):
    """
    Compute the DICE loss with loss mask, similar to generalized IOU for masks
    :param inputs: A float tensor of arbitrary shape. The predictions for each example.
    :param targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
    :param num_masks: Number of valid binary segmentation masks in the batch
    :param loss_mask: A float tensor with the same shape as targets, indicating which
                        pixels (masked patches) contribute to the loss.
    :return: Loss tensor
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # import pdb;pdb.set_trace()
    mask_sum = loss_mask.sum(dim=1)
    # remove loss when
    valid_idx = torch.where(mask_sum)
    valid_mask = loss_mask[valid_idx]
    valid_loss = loss[valid_idx]
    # / valid_mask.sum(1) * mask.shape[-1] is in (0, 1)
    valid_loss_per_sample = valid_loss / valid_mask.sum(1) * loss_mask.shape[-1]

    return valid_loss_per_sample.sum() / num_masks


def sigmoid_ce_loss_with_mask(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        loss_mask: torch.Tensor,
    ):
    """
    Compute the sigmoid cross entropy loss with loss mask
    :param inputs: A float tensor of arbitrary shape. The predictions for each example.
    :param targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
    :param num_masks: Number of valid binary segmentation masks in the batch
    :param loss_mask: A float tensor with the same shape as targets, indicating which
                        pixels (masked patches) contribute to the loss.
    :return: Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    mask_sum = loss_mask.sum(dim=1)
    # remove loss when
    valid_idx = torch.where(mask_sum)
    valid_mask = loss_mask[valid_idx]
    valid_loss = loss[valid_idx]
    # / valid_mask.sum(1) * mask.shape[-1] is in (0, 1)
    valid_loss_per_sample = valid_loss.mean(1) / valid_mask.sum(1) *loss_mask.shape[-1]
    #
    return valid_loss_per_sample.sum() / num_masks

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # import pdb;pdb.set_trace()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def ratio2weight(targets, ratio):
    # import pdb;
    # pdb.set_trace()
    ratio = torch.from_numpy(ratio).type_as(targets)

    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

class MaskedSetLoss(nn.Module):
    """
    Masked Set Loss for segmentation, with uncertainty sampling
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param loss_weight: Weight of the loss
    :param ignore_index: Index to ignore in the loss
    :param loss_per_class: Whether to compute loss per class
    :param num_points: Number of points to sample
    :param oversample_ratio: Oversample ratio of number of points to sample
    :param importance_sample_ratio: Importance sample ratio of number of points to sample
    :param ginfo: Global info
    :param dice_weight: Weight of dice loss
    :param mask_weight: Weight of mask loss
    :param class_weight: Weight of class loss
    :param mask_all_tokens: Whether to mask all tokens, which equals to the original set loss
    """
    def __init__(self, patch_size: int = 16, stride: int = 4,
                 loss_weight = 1, ignore_index=-100,
                 loss_per_class = False, num_points = 12544,
                 oversample_ratio = 3.0, importance_sample_ratio = 0.75,
                 ginfo=None, dice_weight = 5.0, mask_weight = 5.0, class_weight = 2.0,
                 mask_all_tokens=True,aux=False,sample_weight=None,cls_weight_sample=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.ginfo = ginfo
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.class_weight = class_weight
        self.mask_all_tokens = mask_all_tokens

        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_per_class = loss_per_class
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.aux = aux
        self.sample_weight = sample_weight
        self.cls_weight_sample = cls_weight_sample


    def _get_src_permutation_idx(self, indices):
        """permute predictions following indices"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """permute targets following indices"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, inputs, targets, mask=None, modality=''):
        """
        :param inputs: inputs dict, includes pred_masks, pred_logits, etc.
        :param targets: targets dict, includes masks, labels, etc.
        :param mask: mask matrix of targets
        :param modality: modality of the task
        :return: loss_dict, includes dice_loss, mask_loss, class_loss.
        """
        # outputs_without_aux = {k: v for k, v in inputs.items() if k != "aux_outputs"}

        losses = {}
        # pdb.set_trace()
        H, W = targets[0]['masks'].shape[-2:]
        nH, nW = H // self.scale_factor, W // self.scale_factor

        # direct matcher between pred and target
        bs = inputs['pred_masks'].shape[0]
        indices = []
        for b in range(bs):

            tgt_ids = targets[b]["labels"]
            tgt_ids = tgt_ids[tgt_ids!=self.ignore_index]

            row_idx = tgt_ids.cpu().tolist()
            col_idx = list(range(len(tgt_ids)))

            indices.append((row_idx, col_idx))
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # pdb.set_trace()
        src_idx = self._get_src_permutation_idx(matched_indices)
        tgt_idx = self._get_tgt_permutation_idx(matched_indices)

        # whether mask of ignore should be predicted. No, no loss should focus on.
        target_masks = [t["masks"] for t in targets]

        loss_mask = []
        for i in range(len(targets)):
            tgt_ids = targets[i]["labels"]
            tgt_ids = tgt_ids[tgt_ids != self.ignore_index]
            loss_mask.extend([mask[i].unsqueeze(0) for _ in tgt_ids])
        loss_mask = torch.cat(loss_mask,dim=0)

        # Compute the average number of target boxes across all nodes, for normalization purposes

        # num_masks = sum(len(t["labels"])-1*(self.ignore_index in t['labels']) for t in targets)
        num_masks = (loss_mask.sum(1)>0).sum()
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(inputs.values())).device)
        allreduce(num_masks, group_idx=self.ginfo.group)

        target_masks, valid = nested_tensor_from_tensor_list(target_masks).decompose()

        src_masks = inputs['pred_masks'][src_idx]
        # pdb.set_trace()
        target_masks = target_masks[tgt_idx]
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            if self.mask_all_tokens:
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    #loss_mask, nH, nW
                )
            else:
                point_coords = get_uncertain_point_coords_with_randomness_with_mask(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    loss_mask, nH, nW
                )

            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        # pdb.set_trace()
        if self.mask_all_tokens:
            loss_dict = {
                f"loss_{modality}_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks)[0], #* self.mask_weight,
                f"loss_{modality}_dice": dice_loss(point_logits, point_labels, num_masks)[0], #* self.dice_weight,
            }
        else:
            loss_dict = {
                f"loss_{modality}_mask": sigmoid_ce_loss_with_mask(point_logits, point_labels, num_masks, loss_mask)[0],# * self.mask_weight,
                f"loss_{modality}_dice": dice_loss_with_mask(point_logits, point_labels, num_masks, loss_mask)[0],# * self.dice_weight,
            }
        if 'pred_classes' in inputs.keys():
            src_logits = inputs['pred_classes'].float()
            if src_logits.shape[-1] == 1:
                src_logits = src_logits.squeeze(-1)
            target_zeros = torch.zeros_like(src_logits)
            for i, t in enumerate(targets):

                t_ignore = t['labels'][t['labels'] != self.ignore_index]

                target_zeros[i][t_ignore] = 1
            if self.ignore_index>0:
                src_logits = src_logits[:, :self.ignore_index]
                target_zeros = target_zeros[:,:self.ignore_index]
            target = target_zeros.float()
            weight = torch.ones_like(src_logits) * 0.1 # 0.1 for not appear
            weight[target==1]=1
            if self.cls_weight_sample:
                self.sample_weight = np.array(self.sample_weight)
                weight = ratio2weight(target, self.sample_weight)
                loss_ce = F.binary_cross_entropy_with_logits(src_logits, target, weight=weight)
            else:
                loss_ce = F.binary_cross_entropy_with_logits(src_logits, target, weight=weight)

            loss_dict[f'loss_{modality}_bce'] = loss_ce #* self.class_weight
            losses.update(loss_dict)
        if self.aux:
            for idx, aux_inputs in enumerate(inputs["aux_outputs"]):
                H, W = targets[0]['masks'].shape[-2:]
                nH, nW = H // self.scale_factor, W // self.scale_factor

                # direct matcher between pred and target
                bs = aux_inputs['pred_masks'].shape[0]
                indices = []
                for b in range(bs):
                    tgt_ids = targets[b]["labels"]
                    tgt_ids = tgt_ids[tgt_ids != self.ignore_index]

                    row_idx = tgt_ids.cpu().tolist()
                    col_idx = list(range(len(tgt_ids)))

                    indices.append((row_idx, col_idx))
                matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for
                                   i, j in indices]

                src_idx = self._get_src_permutation_idx(matched_indices)
                tgt_idx = self._get_tgt_permutation_idx(matched_indices)

                # whether mask of ignore should be predicted. No, no loss should focus on.
                target_masks = [t["masks"] for t in targets]

                loss_mask = []
                for i in range(len(targets)):
                    tgt_ids = targets[i]["labels"]
                    tgt_ids = tgt_ids[tgt_ids != self.ignore_index]
                    loss_mask.extend([mask[i].unsqueeze(0) for _ in tgt_ids])
                # pdb.set_trace()
                loss_mask = torch.cat(loss_mask, dim=0)

                # Compute the average number of target boxes across all nodes, for normalization purposes

                # num_masks = sum(len(t["labels"])-1*(self.ignore_index in t['labels']) for t in targets)
                num_masks = (loss_mask.sum(1) > 0).sum()
                num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(inputs.values())).device)
                allreduce(num_masks, group_idx=self.ginfo.group)

                target_masks, valid = nested_tensor_from_tensor_list(target_masks).decompose()

                src_masks = aux_inputs['pred_masks'][src_idx]
                target_masks = target_masks[tgt_idx]
                target_masks = target_masks.to(src_masks)

                src_masks = src_masks[:, None]
                target_masks = target_masks[:, None]

                with torch.no_grad():
                    # sample point_coords
                    if self.mask_all_tokens:
                        point_coords = get_uncertain_point_coords_with_randomness(
                            src_masks,
                            lambda logits: calculate_uncertainty(logits),
                            self.num_points,
                            self.oversample_ratio,
                            self.importance_sample_ratio,
                            # loss_mask, nH, nW
                        )
                    else:
                        point_coords = get_uncertain_point_coords_with_randomness_with_mask(
                            src_masks,
                            lambda logits: calculate_uncertainty(logits),
                            self.num_points,
                            self.oversample_ratio,
                            self.importance_sample_ratio,
                            loss_mask, nH, nW
                        )

                    # get gt labels
                    point_labels = point_sample(
                        target_masks,
                        point_coords,
                        align_corners=False,
                    ).squeeze(1)

                point_logits = point_sample(
                    src_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)
                # pdb.set_trace()
                if self.mask_all_tokens:
                    loss_dict = {
                        f"loss_{modality}_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks)[
                                                     0],# * self.mask_weight,
                        f"loss_{modality}_dice": dice_loss(point_logits, point_labels, num_masks)[0],# * self.dice_weight,
                    }
                else:
                    loss_dict = {
                        f"loss_{modality}_mask":
                            sigmoid_ce_loss_with_mask(point_logits, point_labels, num_masks, loss_mask)[
                                0] ,#* self.mask_weight,
                        f"loss_{modality}_dice": dice_loss_with_mask(point_logits, point_labels, num_masks, loss_mask)[
                                                     0] ,#* self.dice_weight,
                    }

                if 'pred_classes' in aux_inputs.keys():
                    src_logits = aux_inputs['pred_classes'].float()
                    if src_logits.shape[-1] == 1:
                        src_logits = src_logits.squeeze(-1)
                    target_zeros = torch.zeros_like(src_logits)
                    for i, t in enumerate(targets):
                        t_ignore = t['labels'][t['labels'] != self.ignore_index]
                        target_zeros[i][t_ignore] = 1
                    if self.ignore_index > 0:
                        src_logits = src_logits[:, :self.ignore_index]
                        target_zeros = target_zeros[:, :self.ignore_index]
                    target = target_zeros.float()
                    weight = torch.ones_like(src_logits) * 0.1  # 0.1 for not appear
                    weight[target == 1] = 1
                    if self.cls_weight_sample:
                        self.sample_weight = np.array(self.sample_weight)
                        weight = ratio2weight(target, self.sample_weight)
                        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target, weight=weight)
                    else:
                        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target, weight=weight)

                    loss_dict[f'loss_{modality}_bce'] = loss_ce #* self.class_weight

                loss_dict = {k + f"_{idx}": v for k, v in loss_dict.items()}
                losses.update(loss_dict)

        return losses
        # return loss_dict



def get_uncertain_point_coords_with_randomness_with_mask(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio, mask, nH, nW
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.

    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    # point_coords = torch.rand(num_boxes, num_sampled, 2, device='cpu') #coarse_logits.device

    # import pdb;pdb.set_trace()
    # mask shape: [samples, nH*nW]
    point_coords = []
    for i in range(mask.shape[0]):
        _mask = mask[i]
        if mask[i].sum()==0:
            # create fake _mask to avoid sampling points from None.
            _mask = torch.zeros_like(mask[i],device=coarse_logits.device)
            _mask[...,-1]=1
        mask_idx = torch.where(_mask)
        # project into the valid space
        mask_valid = _mask[mask_idx]
        mask_valid_length = mask_valid.shape[-1]
        # point_coord = torch.cat((torch.rand(num_sampled, 1)/nH, torch.rand(num_sampled, 1) * mask_valid_length), dim=1)
        point_coord_h = torch.rand(num_sampled, 1, device=coarse_logits.device)/nH
        point_coord_w = torch.rand(num_sampled, 1, device=coarse_logits.device) * mask_valid_length
        try:
            point_coord_w = (mask_idx[0][point_coord_w.long()].float() + point_coord_w % 1)
        except:
            import pdb;pdb.set_trace()
        point_coord_row = point_coord_w // nW
        point_coord_col = point_coord_w % nW

        point_coord_h += point_coord_row * 1/nH
        point_coord_w = point_coord_col * 1/nW

        # project back to the origin space
        point_coord = torch.cat([point_coord_h, point_coord_w], dim=1)
        point_coords.append(point_coord.unsqueeze(0))
    point_coords = torch.cat(point_coords, 0).to(mask.device)
    # pdb.set_trace()


    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


class MaskDetSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses,
                 focal_alpha, ign_thr, ginfo, predict3d=False, xyxy=False,):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ign_thr = ign_thr
        self.ginfo = ginfo
        self.predict3d=predict3d
        self.xyxy=xyxy

    def loss_labels(self, outputs, targets, indices, num_boxes, modality='', log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # import pdb;pdb.set_trace()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        with torch.no_grad():
            src_boxes = outputs['pred_boxes']
            valid_mask = torch.stack([
                torch.all(box_ops.box_ioa(
                    box_ops.box_cxcywh_to_xyxy(boxes),
                    box_ops.box_cxcywh_to_xyxy(target["ignore"])) < self.ign_thr, 1)
                for boxes, target in zip(src_boxes, targets)
            ]) | (target_classes != self.num_classes)

            # ped det never use mask, as it does not need to pred the segmentation mask
            # if outputs['mask'] is not None:
            #     valid_mask &= outputs['mask']['mask']

        src_logits = src_logits[valid_mask]
        target_classes = target_classes[valid_mask]

        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        loss_ce = sigmoid_focal_loss(src_logits, labels, num_boxes, alpha=self.focal_alpha, gamma=2)
        losses = {f'loss_{modality}_ce': loss_ce, "valid_ratio": valid_mask.float().mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['top1'] = accuracy(src_logits, labels)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, modality='',):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses[f'loss_{modality}_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses[f'loss_{modality}_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, modality='', **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, modality=modality, **kwargs)

    def prepare_detection_targets(self, raw_targets):
        new_targets = []
        for targets_per_image in raw_targets:
            # import pdb;pdb.set_trace()
            valid_len = (targets_per_image.area>0).sum()
            iscrowd = targets_per_image.iscrowd[:valid_len]
            boxes = targets_per_image.boxes[:valid_len]
            if self.xyxy:
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

    def forward(self, outputs, raw_targets, mask=None, modality=''):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             raw_targets: list of dicts, such that len(targets) == batch_size. NEED preprocessing!
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             mask: mask matrix for reconstruction
        """
        # import pdb;pdb.set_trace()
        #
        #  use sigmoid as the prediction of det should be in the range of 0,1
        pred_points = outputs['pred_points']
        if self.predict3d:
            outputs['pred_boxes'] = torch.cat([pred_points[:,:,:2], pred_points[:,:,3:5]], dim=-1)
            #  sparse labeling predict boxes is xyxy
        else:
            outputs['pred_boxes'] = pred_points
        if self.xyxy:
            outputs['pred_boxes'] = box_ops.box_xyxy_to_cxcywh(outputs['pred_boxes'])

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        targets = self.prepare_detection_targets(raw_targets.instances)
        # pdb.set_trace()
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        allreduce(num_boxes, group_idx=self.ginfo.group)
        num_boxes = torch.clamp(num_boxes / self.ginfo.task_size, min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, modality, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                pred_points = aux_outputs['pred_points']
                if self.predict3d:
                    aux_outputs['pred_boxes'] = torch.cat([pred_points[:,:,:2], pred_points[:,:,3:5]], dim=-1)
                    #  sparse labing predict boxes is xyxy
                else:
                    aux_outputs['pred_boxes'] = pred_points
                if self.xyxy:
                    aux_outputs['pred_boxes'] = box_ops.box_xyxy_to_cxcywh(aux_outputs['pred_boxes'])

                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, modality, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, modality, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class MaskPOSSetCriterion(nn.Module):
    """This class computes the loss for Pose estimation.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses,
                 ginfo, sample_weight=None, eos_coef=0.1, aux=False, ignore_blank=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer("empty_weight", empty_weight)
        self.sample_weight = np.array(sample_weight)
        # self.valid_channel = sample_weight.nonzero()[0]
        # self.sample_weight = sample_weight[self.valid_channel]

        self.aux = aux
        self.ignore_blank = ignore_blank

        # pointwise mask loss parameters
        # self.num_points = num_points
        # self.oversample_ratio = oversample_ratio
        # self.importance_sample_ratio = importance_sample_ratio
        self.ginfo = ginfo  # distributed info


    def get_accuracy(self, output, target, target_weight):  # for pos
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1x1]):
                Weights across different joint types.
        """

        _, avg_acc, _ = pose_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight.detach().cpu().numpy().squeeze((-2, -1)) > 0)
        accuracy = avg_acc

        return torch.as_tensor([accuracy]).cuda()


    def pos_loss_labels_bce(self, outputs, targets, indices, num_masks):
        """Classification loss (BCE)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_classes" in outputs
        src_logits = outputs["pred_classes"].float()  # [bs, queries, 1]
        if src_logits.shape[-1] == 1:
            src_logits = src_logits.squeeze(-1)

        target_weight = torch.tensor(targets['target_weight']).squeeze()
        target = target_weight.to(outputs["pred_masks"].device)#[:, self.valid_channel]  # remove invalid joints

        if self.sample_weight is not None:
            weight = ratio2weight(target, self.sample_weight)
            loss_ce = F.binary_cross_entropy_with_logits(src_logits, target, weight=weight,)
        else:
            raise NotImplementedError
            weight = torch.ones_like(src_logits)*self.eos_coef
            weight[idx] = 1
            loss_ce = F.binary_cross_entropy_with_logits(src_logits, target,weight=weight)
        losses = {"loss_bce_pos": loss_ce}
        return losses

    def pos_loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        # import pdb;
        # pdb.set_trace()
        _, _, height, width = outputs["pred_masks"].shape
        target_weight = torch.tensor(targets['target_weight'])#[:, self.valid_channel, :]  # (b j 1)
        target_weight = target_weight.unsqueeze(-1)
        target_weight = target_weight.to(outputs["pred_masks"].device)
        criterion = nn.MSELoss(reduction='none')

        if outputs["pred_masks"].shape != targets["label"].shape:
            targets["label"] = F.interpolate(targets["label"], size=(height, width), mode="bilinear")

        loss = criterion(outputs["pred_masks"], targets['label']) * target_weight  # bs, joints, h, w [:, self.valid_channel, :, :]

        losses = {
            "loss_mask_pos": loss.mean(),
            "top1": self.get_accuracy(outputs["pred_masks"], targets["label"], target_weight)
        }
        return losses

    def pos_top1_accuracy(self, loss, outputs, targets, indices, num_masks):
        if outputs["pred_masks"].shape != targets["label"].shape:
            targets["label"] = F.interpolate(targets["label"], size=(height, width), mode="bilinear")

        losses = {
            "top1": self.get_accuracy(outputs["pred_masks"], targets["label"], target_weight)
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'pos_bce_labels': self.pos_loss_labels_bce,
            'pos_mask': self.pos_loss_masks,
            'pos_top1_accuracy': self.pos_top1_accuracy,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, None))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.aux:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, None)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            # "num_points: {}".format(self.num_points),
            # "oversample_ratio: {}".format(self.oversample_ratio),
            # "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
