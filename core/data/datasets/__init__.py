from .images.pedattr_dataset import MultiAttrDataset
from .images.pos_dataset_dev import COCOPosDatasetDev, MPIIPosDatasetDev
from .images.parsing_dataset import (Human3M6ParsingDataset, LIPParsingDataset, CIHPParsingDataset, ATRParsingDataset,
                                     DeepFashionParsingDataset, VIPParsingDataset, ModaNetParsingDataset,
                                     PaperDollParsingDataset)
from .images.multi_posedataset import MultiPoseDatasetDev
from .images.peddet_dataset_v2 import PedestrainDetectionDataset_v2, PedestrainDetectionDataset_v2demo
from .images.image_caption_dataset import CocoCaption, CocoCaptiondemo
from .sequences.skeleton_action_dataset import mmSkeletonDataset
from .images.smpl_dataset_v2 import MeshTSVYamlDataset
from core.utils import printlog

def dataset_entry(config):
    printlog('config[kwargs]',config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
