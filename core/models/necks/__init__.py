from .simple_fpn import (MAEdecoder_proj_neck,)

def neck_entry(config):
    return globals()[config['type']](**config['kwargs'])
