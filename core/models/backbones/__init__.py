# link.nn.SyncNorm is not modified as it is not used in vitdet

from .maskvit import vit_base_patch16_mask, vit_large_patch16_mask

def backbone_entry(config):
    return globals()[config['type']](**config['kwargs'])
