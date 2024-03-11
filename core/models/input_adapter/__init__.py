from .sparse_labeling_adapter import sparse_labeling_adapter, sparse_labeling_adapter_skaction
from .rgb_adapter import rgb_adapter
from .dense_labeling_adapter import dense_labeling_adapter
from .text_adapter import text_adapter, extract_bert_features

def patchembed_entry(config):
    return globals()[config['type']](**config['kwargs'])
