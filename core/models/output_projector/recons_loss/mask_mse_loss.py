import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedMSELoss(nn.Module):
    """
    L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of input image
    :param pix_loss: Whether to use pixel loss
    :param norm_pix_loss: Whether to use normalized pixel loss
    :param pix_loss_weight: Weight of pixel loss
    :param norm_pix_loss_weight: Weight of normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, pix_loss=True,
                 norm_pix_loss=False, pix_loss_weight: float = 1.,
                 norm_pix_loss_weight: float = 1,
                 ginfo = None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.pix_loss = pix_loss
        self.norm_pix_loss = norm_pix_loss
        self.pix_weight = pix_loss_weight
        self.norm_pix_weight = norm_pix_loss_weight
        assert pix_loss or norm_pix_loss

    def patchify(self, imgs, nh, nw):
        """patchify image into patches """
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        """unpatchify patches into images"""
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def loss_with_mask(self, loss, nh, nw, H, W, mask=None):
        """compute loss with mask"""
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            # import pdb
            # pdb.set_trace()

            # to replace nanmean() in the high level pytorch
            valid_mask_idx = torch.where(mask.flatten(start_dim=1).sum(dim=1))
            if torch.isnan(loss[valid_mask_idx].flatten(start_dim=1).sum(dim=1) / mask[valid_mask_idx].flatten(start_dim=1).sum(dim=1)).sum().item()>0:
                import pdb;pdb.set_trace()
            loss = loss[valid_mask_idx].flatten(start_dim=1).sum(dim=1) / mask[valid_mask_idx].flatten(start_dim=1).sum(dim=1)
            loss = loss.mean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training
        return loss

    def forward(self, input, target, mask=None, modality=''):
        """
        :param input: input tensor
        :param target: target images
        :param mask: mask matrix for target images
        :param modality: modality name
        :return: loss dict, including pixel loss and normalized pixel loss
        """
        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        loss_dict = {}
        # import pdb;pdb.set_trace()
        if self.pix_loss:
            loss = F.mse_loss(input, target, reduction='none')
            loss_dict[f'loss_{modality}'] = self.loss_with_mask(loss, nh, nw, H, W, mask) * self.pix_weight

        if self.norm_pix_loss:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)
            loss_norm_pix = F.mse_loss(input, target, reduction='none')
            loss_dict['loss_{}_norm'.format(modality)] = self.loss_with_mask(loss_norm_pix, nh, nw, H, W, mask ) * self.norm_pix_weight

        
        return loss_dict