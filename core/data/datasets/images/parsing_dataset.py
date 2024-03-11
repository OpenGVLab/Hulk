import os
import os.path as osp
import cv2
import torch
import io
import numpy as np
import itertools
from typing import Any, Dict, List, Tuple, Union
from torch.utils import data
from torch.nn import functional as F
from PIL import Image
from core.utils import cv2_loader, pil_loader
from core.data.datasets.images.seg_dataset_dev import Instances, BitMasks
import random
import torch.distributed as dist

import core.data.transforms.parsing_transforms as T
from core.data.transforms.pose_transforms import DataContainer

try:
    from petrel_client.client import Client as Client

    s3client = Client(boto=True,
                      enable_multi_cluster=True,
                      enable_mc=True)
except:
    print("ceph can not be used")
from petrelbox.io import PetrelHelper

palette_dict = {
    'human3m6_parsing': np.array(
        [[0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51], [255, 85, 0], [0, 0, 85], [0, 119, 221],
         [85, 85, 0], [0, 85, 85],
         [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255], [0, 255, 0],
         [51, 170, 221], [0, 255, 255], [255, 170, 85], [85, 255, 170], [170, 85, 52],
         [170, 255, 85], [255, 255, 0], [255, 170, 0], [255, 0, 170], [170, 0, 255]]),
    'LIP_parsing': np.array([[0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
                             [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85,
                                                                       0], [0, 85, 85],
                             [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255],
                             [51, 170, 221], [0, 255, 255], [85, 255, 170], [170, 255, 85],
                             [255, 255, 0], [255, 170, 0]]),
    'CIHP_parsing': np.array([[0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
                              [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85,
                                                                        0], [0, 85, 85],
                              [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255],
                              [51, 170, 221], [0, 255, 255], [85, 255, 170], [170, 255, 85],
                              [255, 255, 0], [255, 170, 0]]),
    'ATR_parsing': np.array([[0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
                             [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85,
                                                                       0], [0, 85, 85],
                             [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255],
                             [51, 170, 221], [0, 255, 255], [85, 255, 170], [170, 255, 85]]),

}

def get_unk_mask_indices(image,num_labels,known_labels,epoch=1,testing=False,):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices

class Human3M6ParsingDataset(data.Dataset):
    task_name = 'human3m6_parsing'
    left_right_pairs = np.array([[1, 6],
                                 [2, 7],
                                 [3, 8],
                                 [17, 25],
                                 [18, 26],
                                 [19, 27],
                                 [33, 38],
                                 [34, 39],
                                 [49, 56],
                                 [50, 58]])

    label_mapper = np.arange(60)

    evaluate_size = (1000, 1000)

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        """human3.6m dataset for human parsing
        Args:
            root_dir ([str]): where dataset
            dataset: train / val
            cfg: yaml format config

            # 0  : background
            # 1  : right hip
            # 2  : right knee
            # 3  : right foot
            # 6  : left hip
            # 7  : left knee
            # 8  : left foot
            # 17 : left shoulder
            # 18 : left elbow
            # 19 : left hand
            # 25 : right shoulder
            # 26 : right elbow
            # 27 : right hand
            # 32 : crotch
            # 33 : right thigh
            # 34 : right calf
            # 38 : left thigh
            # 39 : left calf
            # 43 : lower spine
            # 44 : upper spine
            # 46 : head
            # 49 : left arm
            # 50 : left forearm
            # 56 : right arm
            # 58 : right forearm

        """
        # self.task_name = 'human3m6_parsing'
        self.cfg = cfg
        self.dataset = dataset
        self.is_train = is_train
        self.data_use_ratio = data_use_ratio
        self.pseudo_labels = self.cfg.get('Pseudo_labels', False)
        self.stride_level = self.cfg.get('stride_level', 1)
        # self.palette = palette_dict[self.task_name]
        self.pseudo_labels_palette = palette_dict[self.cfg.get('Pseudo_labels_palette','human3m6_parsing')]
        self.ignore2endclass = self.cfg.get('ignore2endclass', False)

        self.img_list, self.label_list, self.name_list = self._list_dirs(data_path)

        index = np.arange(0, len(self.img_list))
        random.shuffle(index)
        self.img_list = np.array(self.img_list)
        self.label_list = np.array(self.label_list)
        self.name_list = np.array(self.name_list)

        self.img_list = self.img_list[index].tolist()
        self.label_list = self.label_list[index].tolist()
        self.name_list = self.name_list[index].tolist()

        self.images = self.img_list
        self.labels = self.label_list
        self.ignore_label = cfg.ignore_value
        self.num = len(self.images)
        self.num_classes = len(self.cfg.label_list)  # - 1
        assert self.num_classes == self.cfg.num_classes, f"num of class mismatch, len(label_list)={self.num_classes}, num_classes:{self.cfg.num_classes}"

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.original_label = np.array(self.cfg.label_list)

        for i, l in enumerate(self.original_label):
            self.label_mapper[l] = i
        self.mapped_left_right_pairs = self.label_mapper[self.left_right_pairs] if self.left_right_pairs is not None else None

        if self.is_train:
            augs = T.compose([T.hflip(cfg.get("is_flip", False), self.mapped_left_right_pairs),
                              T.resize_image(cfg.crop_size),
                              T.multi_scale(cfg.get("is_multi_scale", False), scale_factor=cfg.get("scale_factor", 11),
                                            center_crop_test=cfg.get("center_crop_test", False),
                                            base_size=cfg.base_size,
                                            crop_size=cfg.crop_size,
                                            ignore_label=cfg.get("ignore_value", 255)),
                              T.rotate(cfg.get("is_rotate", False), degree=cfg.get("degree", 30),
                                       p=cfg.get("possibility", 0.6), pad_val=cfg.get("pad_val", 0),
                                       seg_pad_val=cfg.get("ignore_value", 255)),
                              T.PhotoMetricDistortion(cfg.get('is_photometricdistortion', False),
                                                      brightness_delta=cfg.get('brightness', 32),
                                                      contrast_range=cfg.get('contrast_range', [0.5, 1.5]),
                                                      saturation_range=cfg.get("saturation_range", [0.5, 1.5]),
                                                      hue_delta=cfg.get('hue', 18)
                                                      ),
                              T.transpose()])
        else:
            augs = T.compose([T.resize_image_eval(cfg.eval_crop_size),
                              T.transpose()])
        self.augs = augs

        self.initialized = False
        self.use_ceph = True

    def __len__(self):
        return len(self.img_list)

    def _ignore_to_endclass(self, label):
        label[label==self.ignore_label] = self.num_classes
        return label

    def _read_one(self, index=None):
        if index == None:
            index = np.random.randint(self.num)

        filename = self.img_list[index]
        try:
            img = Image.open(filename).convert('RGB')
            img = np.array(img)[:,:,::-1]

        except:
            outputName = "failed_to_read_in_train.txt"
            with open(outputName, "a") as g:
                g.write("%s\n" % (filename))
            print('Read image[{}] failed ({})'.format(index, filename))
            ## if fail then recursive call _read_one without idx
            return self._read_one()

        gt_label = self.label_list[index]

        try:
            label = np.array(Image.open(gt_label))
        except:
            outputName = "failed_to_read_in_train_labels.txt"
            with open(outputName, "a") as g:
                g.write("%s\n" % (filename))
            print('Read image[{}] failed ({})'.format(index, gt_label))
            ## if fail then recursive call _read_one without idx
            return self._read_one()

        return img, label

    def __getitem__(self, index):

        dataset_dict = {}
        dataset_dict["filename"] = self.name_list[index]

        image, parsing_seg_gt = self._read_one(index)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self._record_image_size(dataset_dict, image)

        if self.pseudo_labels:
            image, parsing_seg_gt = self.augs(image, parsing_seg_gt)
            image = torch.as_tensor(np.ascontiguousarray(image))
            parsing_seg_gt = torch.as_tensor(np.ascontiguousarray(parsing_seg_gt))
            dataset_dict["image"] = image
            dataset_dict['PL_gt'] = parsing_seg_gt
            return dataset_dict

        parsing_seg_gt = self._encode_label(parsing_seg_gt)  # - 1 no need to filter background in human parsing

        size = parsing_seg_gt.size

        if not self.is_train:
            if len(self.evaluate_size) == 2:
                dataset_dict["gt"] = np.copy(
                    cv2.resize(parsing_seg_gt, self.evaluate_size, interpolation=cv2.INTER_LINEAR_EXACT).astype(np.int_))
            else:
                # use DataContainer type to avoid being batched as tensors
                dataset_dict["gt"] = DataContainer(np.copy(parsing_seg_gt.astype(np.int_)))

        parsing_seg_gt = parsing_seg_gt.astype("double")
        assert len(parsing_seg_gt), "parsing needs gt to train"
        image, parsing_seg_gt = self.augs(image, parsing_seg_gt)
        if self.stride_level>1:
            temp_h, temp_w = parsing_seg_gt.shape
            parsing_seg_gt = np.asarray(Image.fromarray(parsing_seg_gt).convert('P').resize((int(temp_h/self.stride_level), int(temp_w/self.stride_level))))
        if self.ignore2endclass:
            parsing_seg_gt = self._ignore_to_endclass(parsing_seg_gt)
        image = torch.as_tensor(np.ascontiguousarray(image))
        parsing_seg_gt = torch.as_tensor(parsing_seg_gt.astype("long"))

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        dataset_dict["image"] = image
        if not self.is_train:
            if self.cfg.get('label_mask', False):
                m = torch.ones(self.num_classes,dtype=torch.int64)*-1 # mask all labels
                dataset_dict['mask'] = m
            return dataset_dict

        dataset_dict["label"] = parsing_seg_gt.long()  # not used in test

        # Prepare per-category binary masks
        parsing_seg_gt = parsing_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = np.unique(parsing_seg_gt)
        # remove ignored region
        if self.cfg.get('add_zero_mask',False):
            classes = np.array(list(range(self.num_classes)))
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        if self.cfg.get('label_mask', False):
            m = np.zeros(self.num_classes)
            m[classes] = 1
            mask = torch.tensor(m, dtype=torch.int64).clone()
            unk_mask_indices = get_unk_mask_indices(image, self.num_classes, known_labels=100,) #  set known_labels>1 to use label masking training
            mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)
            dataset_dict['mask'] = mask

        masks = []
        for class_id in classes:
            masks.append(parsing_seg_gt == class_id)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, parsing_seg_gt.shape[-2], parsing_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances  # not used in test

        return dataset_dict  # {'image': img_mask, 'label': target_mask, 'instances': xxx, 'filename': img_name}

    @staticmethod
    def _record_image_size(dataset_dict, image):
        """
        Raise an error if the image does not match the size specified in the dict.
        """
        # To ensure bbox always remap to original image size
        if "width" not in dataset_dict:
            dataset_dict["width"] = image.shape[1]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.shape[0]

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'eval'
        list_txt = osp.join(data_path, f'flist_2hz_{train_type}.txt')

        # with open(list_txt, 'r') as f:
        #     data = f.readlines()
        # data = [d.strip() for d in data]
        list_lines = s3client.Get(list_txt)
        if not list_lines:
            print('File not exist', list_txt)
            import pdb;
            pdb.set_trace()
            raise IOError('File not exist', list_file)
        list_lines = list_lines.decode('ascii')
        data = list_lines.split('\n')
        data = [d for d in data if len(d)]


        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        for d in data:
            img_path = osp.join(data_path, d)
            image_name = '/'.join(d.split('/')[2:])
            label_path = img_path.replace('rgb', 'seg', 1)

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        return img_list, label_list, name_list

    def _encode_label(self, labelmap):
        shape = labelmap.shape
        encoded_labelmap = np.zeros(shape=(shape[0], shape[1]), dtype=np.uint8)
        for i, class_id in enumerate(self.cfg.label_list):
            encoded_labelmap[labelmap == class_id] = i

        return encoded_labelmap

    def __repr__(self):
        return self.__class__.__name__ + \
               f'rank: {self.rank} task: {self.task_name} mode:{"training" if self.is_train else "inference"} ' \
               f'dataset_len:{len(self.img_list)} id_num:{self.cfg["num_classes"]} augmentation: {self.augs}'

class LIPParsingDataset(Human3M6ParsingDataset):
    """
    0:'background',
    1:'hat',
    2:'hair',
    3:'glove',
    4:'sunglasses',
    5:'upperclothes',
    6:'dress',
    7:'coat',
    8:'socks',
    9:'pants',
    10:'jumpsuits',
    11:'scarf',
    12:'skirt',
    13:'face',
    14:'leftArm',
    15:'rightArm',
    16:'leftLeg',
    17:'rightLeg',
    18:'leftShoe',
    19:'rightShoe'
    """
    task_name = 'LIP_parsing'

    left_right_pairs = np.array(
        [[14, 15], [16, 17], [18, 19]]
    )

    label_mapper = np.arange(60)

    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(LIPParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,
                                                data_use_ratio=data_use_ratio,
                                                dataset=dataset, is_train=is_train,
                                                cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'
        """
        - LIP
            -data
                -train_id.txt
                -train_images
                    -1000_1234574.jpg
                -val_images
                -val_id.txt
            -Trainval_parsing_annotations
                -train_segmentations
                    -1000_1234574.png
        """
        list_txt = osp.join(data_path, 'data', f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        postfix_img = '.jpg'
        postfix_ann = '.png'
        for d in data:
            img_path = osp.join(data_path, f'data/{train_type}_images', d + postfix_img)
            image_name = d
            label_path = osp.join(data_path, f'TrainVal_parsing_annotations/{train_type}_segmentations',
                                  d + postfix_ann)

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        return img_list, label_list, name_list

class CIHPParsingDataset(Human3M6ParsingDataset):
    """
    0:'background',
    1:'hat',
    2:'hair',
    3:'glove',
    4:'sunglasses',
    5:'upperclothes',
    6:'dress',
    7:'coat',
    8:'socks',
    9:'pants',
    10:'torsoSkin',
    11:'scarf',
    12:'skirt',
    13:'face',
    14:'leftArm',
    15:'rightArm',
    16:'leftLeg',
    17:'rightLeg',
    18:'leftShoe',
    19:'rightShoe'
    """
    task_name = 'CIHP_parsing'

    left_right_pairs = np.array(
        [[14, 15], [16, 17], [18, 19]]
    )

    label_mapper = np.arange(60)

    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(CIHPParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,data_use_ratio=data_use_ratio,
                                                 dataset=dataset, is_train=is_train,
                                                 cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'
        """
        - CHIP
            -instance-level_human_parsing
                -Training
                    -Images
                        -0008522.jpg
                    -Category_ids
                        -0008522.png
                    -train_id.txt
                -Validation
                    -val_id.txt
        """
        Infix = 'Training' if train_type == 'train' else 'Validation'
        list_txt = osp.join(data_path, 'instance-level_human_parsing', Infix, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]

        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        postfix_img = '.jpg'
        postfix_ann = '.png'
        for d in data:
            img_path = osp.join(data_path, 'instance-level_human_parsing', Infix, f'Images', d + postfix_img)
            image_name = d
            label_path = osp.join(data_path, 'instance-level_human_parsing', Infix, 'Category_ids', d + postfix_ann)

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        return img_list, label_list, name_list


class ATRParsingDataset(Human3M6ParsingDataset):
    """
    0:'background', #
    1:'hat', #
    2:'hair',#
    3:'sunglasses',#
    4:'upperclothes',#
    5:'skirt',
    6:'pants',#
    7:'dress',#
    8:'belt',
    9:'leftshoe',#
    10:'rightshoe',#
    11:'face',#
    12:'leftleg',#
    13:'rightleg',#
    14:'leftarm',#
    15:'rightarm',#
    16:'bag',#
    17:'scarf',#
    """
    task_name = 'ATR_parsing'

    left_right_pairs = np.array(
        [[9,10], [12,13], [14,15]]
    )

    label_mapper = np.arange(60)

    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(ATRParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,
                                                data_use_ratio=data_use_ratio,
                                                dataset=dataset, is_train=is_train,
                                                cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'
        """
        - ATR
            -humanparsing
                -JPEGImages
                -SegmentationClassAug
            -train_id.txt
            -val_id.txt
        """
        list_txt = osp.join(data_path, f'{train_type}_id.txt')
        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]

        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        postfix_img = '.jpg'
        postfix_ann = '.png'
        for d in data:
            img_path = osp.join(data_path, f'humanparsing/JPEGImages', d + postfix_img)
            image_name = d
            label_path = osp.join(data_path, f'humanparsing/SegmentationClassAug',
                                  d + postfix_ann)

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        return img_list, label_list, name_list


class DeepFashionParsingDataset(Human3M6ParsingDataset):
    """
    0:'background', #
    1:'hat', #
    2:'hair',#
    3:'sunglasses',#
    4:'upperclothes',#
    5:'skirt',
    6:'pants',#
    7:'dress',#
    8:'belt',
    9:'leftshoe',#
    10:'rightshoe',#
    11:'face',#
    12:'leftleg',#
    13:'rightleg',#
    14:'leftarm',#
    15:'rightarm',#
    16:'bag',#
    17:'scarf',#
    """
    task_name = 'DeepFashion_parsing'
    label_mapper = np.arange(60)
    left_right_pairs = None
    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(DeepFashionParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,
                                                        data_use_ratio=data_use_ratio,
                                                        dataset=dataset, is_train=is_train,
                                                        cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'
        """
        - DeepFashion
            -humanparsing
                -JPEGImages
                -SegmentationClassAug
            -train_id.txt
            -val_id.txt
        """
        list_txt = osp.join(data_path, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        postfix_img = '.jpg'
        postfix_ann = '.png'

        if train_type == 'train':
            for d in data:
                img_path = osp.join(data_path, f'train/image', d + postfix_img)
                image_name = d
                label_path = osp.join(data_path, f'train/seg',
                                      d + postfix_ann)

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

            return img_list, label_list, name_list
        else:
            raise ValueError("not implement")


class VIPParsingDataset(Human3M6ParsingDataset):
    task_name = 'VIP_parsing'
    left_right_pairs = np.array(
        [[14, 15], [16, 17], [18, 19]]
    )

    label_mapper = np.arange(60)

    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(VIPParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,data_use_ratio=data_use_ratio,
                                                dataset=dataset, is_train=is_train,
                                                cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'

        list_txt = osp.join(data_path, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        postfix_img = '.jpg'
        postfix_ann = '.png'

        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        if train_type == 'train':
            for d in data:
                img_path = osp.join(data_path, f'Images', d + postfix_img)
                image_name = d
                label_path = osp.join(data_path, f'Annotations/Category_ids',
                                      d + postfix_ann)

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

            return img_list, label_list, name_list
        else:
            raise ValueError("not implement")

class PaperDollParsingDataset(Human3M6ParsingDataset):
    """
    0:'background',
    1:'hat',
    2:'hair',
    3:'glove',
    4:'sunglasses',
    5:'upperclothes',
    6:'dress',
    7:'coat',
    8:'socks',
    9:'pants',
    10:'torsoSkin',
    11:'scarf',
    12:'skirt',
    13:'face',
    14:'leftArm',
    15:'rightArm',
    16:'leftLeg',
    17:'rightLeg',
    18:'leftShoe',
    19:'rightShoe'
    """
    task_name = 'PaperDoll_parsing'

    left_right_pairs = np.array(
        [[14, 15], [16, 17], [18, 19]]
    )

    label_mapper = np.arange(60)

    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 data_use_ratio=1,
                 dataset='train',
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(PaperDollParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path, data_use_ratio=data_use_ratio,
                                                      dataset=dataset, is_train=is_train,
                                                      cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'
        """
        - PaperDoll_folder
            - TrainVal_parsing_annotations/
                - 0000000.png
            - images
                - 0000000.jpg
        """
        list_txt = osp.join(data_path, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        postfix_img = '.jpg'
        postfix_ann = '.png'

        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        for d in data:
            img_path = osp.join(data_path, 'images', d + postfix_img)
            image_name = d
            label_path = osp.join(data_path, 'TrainVal_parsing_annotations/', d + postfix_ann)

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        return img_list, label_list, name_list


class FashionPediaParsingDataset(Human3M6ParsingDataset):
    task_name = 'FashionPedia_parsing'

    label_mapper = np.arange(60)

    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 data_use_ratio=1,
                 dataset='train',
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(FashionPediaParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,
                                                         data_use_ratio=data_use_ratio,
                                                         dataset=dataset, is_train=is_train,
                                                         cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'

        list_txt = osp.join(data_path, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        postfix_img = '.jpg'
        postfix_ann = '.png'

        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        if train_type == 'train':
            for d in data:
                img_path = osp.join(data_path, 'train/', d + postfix_img)
                image_name = d
                label_path = osp.join(data_path, 'train_annotation/', d + postfix_ann)

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

            return img_list, label_list, name_list

        elif train_type == 'val':
            for d in data:
                img_path = osp.join(data_path, 'test/', d + postfix_img)
                image_name = d
                label_path = osp.join(data_path, 'test_annotation/', d + postfix_ann)

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

            return img_list, label_list, name_list
        else:
            raise


class ModaNetParsingDataset(Human3M6ParsingDataset):
    """
    modanet_par = {
    0: 'Background',
    1: 'Bag',
    2: 'Belt',
    3: 'Boots',
    4: 'Footwear',
    5: 'Outer',
    6: 'Dress',
    7: 'Sunglasses',
    8: 'Pants',
    9: 'Top',
    10: 'Shorts',
    11: 'Skirt',
    12: 'Headwear',
    13: 'Scarf & Tie'
}
    """

    task_name = 'ModaNet_parsing'
    label_mapper = np.arange(60)
    left_right_pairs = None
    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(ModaNetParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,data_use_ratio=data_use_ratio,
                                                    dataset=dataset, is_train=is_train,
                                                    cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()
        # image_dir = osp.join(data_path, 'protocal_1', 'rgb')
        # label_dir = osp.join(data_path, 'protocal_1', 'seg')

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'

        list_txt = osp.join(data_path, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))

        postfix_img = '.jpg'
        postfix_ann = '.png'

        if train_type == 'train':
            for d in data:
                img_path = osp.join(data_path, f'images', d + postfix_img)
                image_name = d
                label_path = osp.join(data_path, f'seg',
                                      d + postfix_ann)

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

            return img_list, label_list, name_list
        else:
            raise ValueError("not implement")


class MHPParsingDataset(Human3M6ParsingDataset):
    task_name = 'MHP_parsing'
    label_mapper = np.arange(60)
    left_right_pairs = None
    evaluate_size = ()

    def __init__(self,
                 ginfo,
                 data_path,
                 dataset='train',
                 data_use_ratio=1,
                 is_train=True,
                 cfg=None,
                 **kwargs):
        super(ModaNetParsingDataset, self).__init__(ginfo=ginfo, data_path=data_path,data_use_ratio=data_use_ratio,
                                                    dataset=dataset, is_train=is_train,
                                                    cfg=cfg, **kwargs)

    def _list_dirs(self, data_path):
        img_list = list()
        label_list = list()
        name_list = list()

        if self.dataset == 'train':
            train_type = 'train'
        elif self.dataset == 'val':
            train_type = 'val'

        list_txt = osp.join(data_path, f'{train_type}_id.txt')

        with open(list_txt, 'r') as f:
            data = f.readlines()
        data = [d.strip() for d in data]


        if self.data_use_ratio != 1:
            data = random.sample(data, int(len(data) * self.data_use_ratio))
            
        postfix_img = '.jpg'
        postfix_ann = '.png'

        if train_type == 'train':
            for d in data:
                img_path = osp.join(data_path, f'images/', d + postfix_img)
                image_name = d
                label_path = osp.join(data_path, f'processed_label/',
                                      d + postfix_ann)

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

            return img_list, label_list, name_list
        else:
            raise ValueError("not implement")

