from torch.nn import CrossEntropyLoss, MSELoss
from typing import Callable, Optional
from torch import Tensor
import torch. nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer
LABLE_SMOOTHING = 'default'
if '2.0' in torch.__version__:
    LABLE_SMOOTHING = '2.0'

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

# Label smoothing
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0., reduction='mean', cfg=None, loss_weight=1., exp_sample_weight=False,
                 patch_size=None, stride=None, ginfo=None, sample_weight_path='', ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.tokenizer = BertTokenizer('./bert-base-uncased/vocab.txt')
        self.ignore_index = ignore_index
        self.sample_weight_path = sample_weight_path
        self.exp_sample_weight = exp_sample_weight

        self.sample_weight = None


    def forward(self, preds, target, mask=None, modality='text'):

        if LABLE_SMOOTHING == '2.0':
            loss_out = F.cross_entropy(preds.flatten(0, 1), target[:, 1:].flatten(),
                                       weight=self.sample_weight.cuda() if self.sample_weight is not None else None,
                                       reduction=self.reduction,
                                       label_smoothing=self.epsilon,
                                       ignore_index=self.ignore_index)

            valid = target[:, 1:]!= 0
            pred = torch.argmax(preds, -1)
            pred_valid = pred[valid]
            target_valid = target[:, 1:][valid]
            valid_top1 = torch.sum(pred_valid == target_valid).float() / torch.sum(valid)
            return {'loss': loss_out, 'top1': valid_top1}

        else:
            preds = preds.flatten(0, 1).float()
            num = preds.size()[-1]
            log_preds = F.log_softmax(preds, dim=-1)
            loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
            nll = F.nll_loss(log_preds, target[:, 1:].flatten(), reduction=self.reduction)
            loss_out = linear_combination(loss/num, nll, self.epsilon) * self.loss_weight

            valid = target[:, 1:] != 0
            pred = torch.argmax(preds, -1)
            pred_valid = pred[valid.flatten(0,1)]
            target_valid = target[:, 1:][valid]
            top1 = torch.sum(pred_valid == target_valid).float() / torch.sum(valid)

        return {f'loss_{modality}': loss_out, 'top1': top1}

