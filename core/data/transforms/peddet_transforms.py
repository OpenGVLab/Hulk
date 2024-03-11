# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import os.path as osp
import json
import torch
import torch.utils.data
import torchvision
import core.data.transforms.peddet_transforms_helpers.transforms as T
import cv2
cv2.ocl.setUseOpenCL(False)

class PedestrainDetectionAugmentation(object):
    def __init__(self, phase, vit=False, return_box_xyxy=False, max_size=1333,):
        if vit:
            normalize = T.Compose([
                T.PILToTensor(),
                T.Normalize([0., 0., 0.], [1., 1., 1.], return_box_xyxy=return_box_xyxy)
            ])
        else:
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], return_box_xyxy=return_box_xyxy)
            ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

        if phase == 'train':
            self.transformer = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
            ])
        elif phase == 'val':
            self.transformer = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])
        else:
            raise NotImplementedError

    def __call__(self, image, target):
        return self.transformer(image, target)
