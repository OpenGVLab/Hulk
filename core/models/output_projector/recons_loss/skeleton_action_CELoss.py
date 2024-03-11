import torch
import torch.nn as nn
from core.utils import accuracy

class CELoss(nn.Module):
    def __init__(self, loss_weight=1.0, **kwargs):
        super(CELoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, logits, label, mask=None, modality='text'):
        if len(label.size()) > 1:
            label = torch.argmax(label, dim=1)
        ce_loss = self.ce(logits, label) * self.loss_weight
        top1 = accuracy(logits.data, label.cuda(), topk=(1, 5))[0]
        return {f'loss_{modality}_CE': ce_loss, 'top1': top1}