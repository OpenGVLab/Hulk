import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from core.data.transforms.skeleton_transforms import *
import pickle
import math
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Sequence

class mmSkeletonDataset(Dataset):
    def __init__(self,
                 ginfo,
                 ann_file,
                 dataset_name,
                 kp_dim,
                 num_classes,
                 data_pipeline=None,
                 one_hot=False,
                 split='train',
                 data_prefix='',
                 select_data=False,
                 centernorm=False,
                 flip=False,
                 scale_range=False,
                 **kwargs):
        self.task_name = ginfo.task_name
        self.split = split
        self.ann_file = ann_file
        self.dataset_name = dataset_name
        self.data_prefix = data_prefix
        self.num_classes = num_classes
        self.total_sk_begin = []
        self.total_sk_end = []
        self.select_data = select_data
        self.centernorm=centernorm
        self.flip = flip
        self.scale_range = scale_range

        base = 0
        for this_num_class in num_classes:
            self.total_sk_begin.append(base)
            self.total_sk_end.append(base + this_num_class)
            base = base + this_num_class
        assert self.total_sk_end[-1] == sum(num_classes)

        self.sample_cls_begin = []
        self.sample_cls_end = []

        self.kp_dim = kp_dim
        self.one_hot = one_hot
        self.left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
        self.right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
        self.video_infos = self.load_pkl_annotations()

        pipeline = [globals()[op['type']](**op['kwargs']) for op in data_pipeline]

        self.pipeline = ComposeX(pipeline)


    def load_pkl_annotations(self):
        """load the data"""
        if isinstance(self.ann_file, list):
            merge_data = []
            for i, single_pkl in enumerate(self.ann_file):
                with open(single_pkl, 'rb') as f:
                    data = pickle.load(f)
                dataset_name = self.dataset_name[i]
                if self.split == 'train':
                    if dataset_name == 'ntu60':  # ['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                        dataset_split = 'xsub_train'
                    elif dataset_name == 'ntu120':  # ['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                        dataset_split = 'xsub_train'
                    elif dataset_name == '2dntu60':  # ['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                        dataset_split = 'xsub_train'
                    elif dataset_name == '2dntu120':  # ['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                        dataset_split = 'xsub_train'
                    elif dataset_name == 'gym':  # ['train', 'val']
                        dataset_split = 'train'
                    elif dataset_name == 'diving':  # ['train', 'val']
                        dataset_split = 'train'
                    elif dataset_name == 'ucf':  # ['train1', 'val']
                        dataset_split = 'train1'
                    elif dataset_name == 'k400':  # ['train', 'val']
                        dataset_split = 'train'
                elif self.split == 'test':
                    if dataset_name == 'ntu60': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                        dataset_split = 'xsub_val'
                    elif dataset_name == 'ntu120': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                        dataset_split = 'xsub_val'
                    elif dataset_name == '2dntu60': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                        dataset_split = 'xsub_val'
                    elif dataset_name == '2dntu120': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                        dataset_split = 'xsub_val'
                    elif dataset_name == 'gym': #['train', 'val']
                        dataset_split = 'val'
                    elif dataset_name == 'diving': #['train', 'val']
                        dataset_split = 'test'
                    elif dataset_name == 'ucf': #['train', 'val']
                        dataset_split = 'test1'
                    elif dataset_name == 'k400': #['train', 'val']
                        dataset_split = 'val'

                split, data = data['split'], data['annotations']
                identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
                split = set(split[dataset_split])
                data = [x for x in data if x[identifier] in split]
                # modify the label value
                if i > 0:
                    for x in data:
                        x['label'] += sum(self.num_classes[:i])

                if self.select_data is not False and self.select_data != dataset_name and self.split == 'test': continue
                merge_data.extend(data)

                self.sample_cls_begin.extend([self.total_sk_begin[i] for _ in range(len(data))])
                self.sample_cls_end.extend([self.total_sk_end[i] for _ in range(len(data))])

            data = merge_data

        else: # single pkl for single dataset in train, and for all test (only one dataset)

            with open(self.ann_file, 'rb') as f:
                data = pickle.load(f)

            if self.split == 'train':
                if self.dataset_name == 'ntu60': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                    dataset_split = 'xsub_train'
                elif self.dataset_name == 'ntu120': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                    dataset_split = 'xsub_train'
                elif self.dataset_name == '2dntu60':  # ['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                    dataset_split = 'xsub_train'
                elif self.dataset_name == '2dntu120':  # ['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                    dataset_split = 'xsub_train'
                elif self.dataset_name == 'gym': #['train', 'val']
                    dataset_split = 'train'
                elif self.dataset_name == 'diving': #['train', 'val']
                    dataset_split = 'train'
                elif self.dataset_name == 'ucf': #['train', 'val']
                    dataset_split = 'train1'
                elif self.dataset_name == 'k400': #['train', 'val']
                    dataset_split = 'train'
                # for ucf hmdb, use all data, including train and  test...

            elif self.split == 'test':
                if self.dataset_name == 'ntu60': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                    dataset_split = 'xsub_val'
                elif self.dataset_name == 'ntu120': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                    dataset_split = 'xsub_val'
                elif self.dataset_name == '2dntu60': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val', 'xview_train', 'xview_val']
                    dataset_split = 'xsub_val'
                elif self.dataset_name == '2dntu120': #['xsub_train', 'xsub_val', 'xset_train', 'xset_val']
                    dataset_split = 'xsub_val'
                elif self.dataset_name == 'gym': #['train', 'val']
                    dataset_split = 'val'
                elif self.dataset_name == 'diving': #['train', 'val']
                    dataset_split = 'test'
                elif self.dataset_name == 'ucf': #['train', 'val']
                    dataset_split = 'test1'
                elif self.dataset_name == 'k400': #['train', 'val']
                    dataset_split = 'val'

            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[dataset_split])
            data = [x for x in data if x[identifier] in split]

        return data

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['index'] = idx
        results['modality'] = "Pose"
        results['test_mode'] = True if self.split == 'test' else False

        out = self.pipeline(results)
        kp = out.pop('keypoint')
        if self.centernorm:
            if 'ntu60' in self.dataset_name:
                kp = CenterNorm_25j(kp)
            else:
                raise NotImplementedError('only support ntu60 for 3d skeleton')
                kp = CenterNorm_17j(kp)
        if self.split == 'train':
            if self.scale_range:
                kp = Scale(kp,self.scale_range)
            if self.flip:
                kp = Flip3d(kp)
        else:
            pass
        out['sparse_labeling'] = kp

        if self.one_hot:
            label = out.pop('label')
            if isinstance(self.num_classes,list):
                out['label'] = np.zeros(sum(self.num_classes), dtype=np.float32) - 1
            else:
                out['label'] = np.zeros(self.num_classes, dtype=np.float32) - 1
            out['label'][self.sample_cls_begin[idx]: self.sample_cls_end[idx]] = 0.
            out['label'][label] = 1
        else:
            pass

        out['attr_begin'] = self.sample_cls_begin[idx]
        out['attr_end'] = self.sample_cls_end[idx]

        return out  

def vertices_3dembed(vertices, num_vertices_feats=256, temperature=10000):
    """
    project vertices to 3d embedding
    :param vertices:
    :param num_vertices_feats:
    :param temperature:
    :return:
    """
    scale = 2 * math.pi
    vertices = vertices * scale
    dim_t = torch.arange(num_vertices_feats, dtype=torch.float32, device='cpu')
    dim_t = temperature ** (2 * (dim_t // 2) / num_vertices_feats)
    vertices_x = vertices[..., 0, None] / dim_t  # QxBx128
    vertices_y = vertices[..., 1, None] / dim_t
    vertices_z = vertices[..., 2, None] / dim_t

    vertices_x = torch.stack((vertices_x[..., 0::2].sin(), vertices_x[..., 1::2].cos()), dim=-1).flatten(-2)
    vertices_y = torch.stack((vertices_y[..., 0::2].sin(), vertices_y[..., 1::2].cos()), dim=-1).flatten(-2)
    vertices_z = torch.stack((vertices_z[..., 0::2].sin(), vertices_z[..., 1::2].cos()), dim=-1).flatten(-2)

    verticesemb = torch.cat((vertices_z, vertices_y, vertices_x), dim=-1)
    del vertices_x, vertices_y, vertices_z, vertices, dim_t
    return verticesemb


