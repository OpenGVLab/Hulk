from .rgb_projector import rgb_projector
from .dense_labeling_projector import dense_labeling_projector
from .text_projector import text_projector
from .sparse_labeling_projector import sparse_labeling_projector


def outputproj_entry(config):
    return globals()[config['type']](**config['kwargs'])