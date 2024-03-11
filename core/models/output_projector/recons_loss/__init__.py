from .mask_mse_loss import MaskedMSELoss
from .mask_cross_entropy_loss import MaskedCrossEntropyLoss
from .mask_set_loss import MaskedSetLoss
from .L1_loss import L1Loss
from .mask_attr_loss import MaskedOneSideBCELoss
from .skeleton_action_CELoss import CELoss
from .smpl_losses_fastmetro import SMPL_LOSS_FASTMETRO 
from .mask_det_set_loss import MaskDetFocalDiceLoss
from .mask_pose_set_loss import POS_FocalDiceLoss_bce_cls_emb, POSDiceLoss
from .mask_parsing_set_loss import FocalDiceLoss_bce_cls_emb_sample_weight
from .caption_loss import LabelSmoothingCrossEntropy

def recons_loss_entry(config):
    return globals()[config['type']](**config['kwargs'])
