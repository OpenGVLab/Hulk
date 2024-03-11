import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param label_smoothing: Amount of smoothing in the loss (default is 0.0)
    :param loss_weight: Weight of the loss (default is 1.0)
    :param ignore_index: Index to ignore in the loss (default is -100)
    :param loss_per_class: Whether to compute loss per class (default is False)
    :param loss_backgroud_weight: Weight of the background class (default is 1.0)
    :param ginfo: Global info
    """

    def __init__(self, patch_size: int = 16, stride: int = 4,
                 label_smoothing : float = 0.0, loss_weight = 1, ignore_index=-100,
                 loss_per_class = False, loss_backgroud_weight = 1, ginfo=None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        # self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_per_class = loss_per_class
        self.loss_background_weight = loss_backgroud_weight

    def forward(self, input, target, mask=None, modality=''):
        """
        :param input: input tensor
        :param target: target labels
        :param mask: mask matrix
        :param modality: modality name of the loss
        :return:
        """
        loss_dict = {}
        loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            H, W = input.shape[-2:]
            nh, nw = H // self.scale_factor, W // self.scale_factor
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            if self.loss_per_class:
                for idx in range(self.ignore_index):
                    if mask[target==idx].shape[0]:
                        # mask_ = mask*(target==idx)
                        loss_ = loss*(target==idx)
                        loss__ = loss_.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
                        loss_dict[f'loss_{modality}_{idx}'] = loss__.mean() if idx!=0 else loss__.mean()*self.loss_background_weight
                    else:
                        loss_dict[f'loss_{modality}_{idx}'] = loss.mean() * 0
            else:
                loss = loss * mask
                # Compute mean per sample
                loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
                loss = loss.mean()  # Account for zero masks
            # import pdb;pdb.set_trace()
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training
        if not self.loss_per_class:
            loss_dict[f'loss_{modality}'] = loss*self.loss_weight
        # pdb.set_trace()
        return loss_dict



