import torch.distributed as dist
from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam, AdamW # noqa F401
from .lars import LARS  # noqa F401
from .adam_clip import AdamWithClip, AdamWWithClip, AdamWWithClipDev, AdamWWithBackboneClipDev  # noqa F401
from .adafactor import Adafactor_dev

# FusedFP16SGD is not used when training vits
FusedFP16SGD = None

def optim_entry(config):
    rank = dist.get_rank()
    if config['type'] == 'FusedFP16SGD' and FusedFP16SGD is None:
        raise RuntimeError('FusedFP16SGD is disabled due to linklink version, try using other optimizers')
    if config['type'] == 'FusedFP16SGD' and rank > 0:
        config['kwargs']['verbose'] = False
    return globals()[config['type']](**config['kwargs'])
