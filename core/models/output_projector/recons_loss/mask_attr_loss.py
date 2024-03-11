import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ratio2weight(targets, ratio):
    """
    Compute the weights for each sample in a batch based on the ratio.
    :param targets: target labels of a batch
    :param ratio: computed ratio for each attribute
    :return: weights for each sample in a batch
    """
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

def focal_weight(inputs, targets, gamma: float = 2):
    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return focal_weight

def ratio2weight_CEloss_changed_to_BCEloss(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    weights = torch.exp(-torch.log(ratio) * targets)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must greater than 1
    weights[targets > 1] = 0.0

    return weights

class MaskedOneSideBCELoss(nn.Module):
    """
    Masked One Side BCE Loss. Only consider the positive language meannings
    :param loss_weight: loss weight
    :param sample_weight: sample weight
    """
    def __init__(self, loss_weight=1.,
                 sample_weight=None,
                 dataset_weight=None,
                 patch_size=[None,None],  # placeholder, achieve similar kwargs with other adapters
                 stride=None,
                 ginfo=None,
                 cal_acc=False,
                 celoss_changed_to_bceloss=False,
                 use_focal_weight=False,
                 ):
        super(MaskedOneSideBCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.sample_weight = sample_weight
        self.dataset_weight = dataset_weight
        self.cal_acc = cal_acc
        self.celoss_changed_to_bceloss = celoss_changed_to_bceloss
        self.use_focal_weight = use_focal_weight

        if sample_weight is not None:
            self.sample_weight = np.array(self.sample_weight)
        if dataset_weight is not None:
            self.dataset_weight = np.array(dataset_weight)

    def forward(self, logits, targets, mask=None, modality='text'):
        """
        :param logits: predicted logits
        :param targets: target labels of a batch
        :param mask: mask matrix on targets
        :param modality: modality name of loss
        :return: loss_dict: loss dict
        """
        batch_size = logits.shape[0]
        weight_mask = (targets != -1)  # mask -1 labels from HARDHC dataset
        if mask is not None:
            weight_mask = weight_mask * mask    # mask out the random masking part
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', weight=weight_mask) * self.loss_weight
        # import pdb;pdb.set_trace()
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            if self.celoss_changed_to_bceloss:
                weight = torch.where(targets.detach().cpu()>0.5, torch.ones(1)*60, torch.ones(1))
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            if self.use_focal_weight:
                focalweight = focal_weight(logits, targets)
                loss = (loss * weight.cuda() * focalweight.cuda())
                del weight, targets_mask, weight_mask, focalweight
            else:
                loss = (loss * weight.cuda())
                del weight, targets_mask, weight_mask
            if self.dataset_weight is not None:
                dataset_weight = torch.from_numpy(self.dataset_weight).type_as(loss)
                loss = loss * dataset_weight.cuda()
                del dataset_weight
            torch.cuda.empty_cache()

        loss_dict = {f'loss_{modality}_OSBCE': loss.sum() / batch_size}
        if self.cal_acc:
            #  now only for skeleton action recognition
            #  change one-hot label to label index
            targets = torch.argmax(targets, dim=1)
            with torch.no_grad():
                prec1, prec5 = accuracy(logits.data, targets, topk=(1, 5))
            loss_dict['top1'] = prec1
            del targets, prec1, prec5
            torch.cuda.empty_cache()
        del loss, logits
        torch.cuda.empty_cache()

        return loss_dict

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    del pred, correct, correct_k
    torch.cuda.empty_cache()
    return res