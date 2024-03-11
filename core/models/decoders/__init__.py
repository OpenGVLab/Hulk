from .mask2former import UniHCPv2_Head


def decoder_entry(config):
    return globals()[config['type']](**config['kwargs'])
