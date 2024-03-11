import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import logging
import numpy as np
from copy import deepcopy
from .seg_tester_dev import DatasetEvaluator

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.optim
import torch.multiprocessing as mp
import torch.utils.data

class SkeletonActionEvaluator(DatasetEvaluator):

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):

        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

    def reset(self):
        self.gt_label = []
        self.preds_probs = []

    def process(self, inputs, outputs):
        gt_label = inputs['label']
        preds_probs = outputs['logit']
        self.gt_label.append(gt_label)
        self.preds_probs.append(preds_probs)

    @staticmethod
    def all_gather(data, group=0):
        assert dist.get_world_size() == 1, \
            f"distributed eval unsupported yet, \
            uncertain if we can use torch.dist \
            with link jointly"
        if dist.get_world_size() ==1:
            return [data]

        world_size = dist.get_world_size()
        tensors_gather = [torch.ones_like(data) for _ in range(world_size)]
        dist.allgather(tensors_gather, data, group=group)
        return tensors_gather

    def evaluate(self):
        gt_label = torch.cat(self.gt_label, dim=0)
        preds_probs = torch.cat(self.preds_probs, dim=0)

        if self._distributed:
            torch.cuda.synchronize()

            gt_label = self.all_gather(gt_label)
            preds_probs = self.all_gather(preds_probs)

            if dist.get_rank() != 0:
                return

        gt_label = torch.cat(gt_label, dim=0)
        preds_probs = torch.cat(preds_probs, dim=0)
        if gt_label.shape[0] != preds_probs.shape[0]:
            bl = gt_label.shape[0]
            bp = preds_probs.shape[0]
            num_clip = int(bp/bl)
            gt_label = gt_label.unsqueeze(1).repeat(1,num_clip).reshape(-1)
        prec1, prec5 = accuracy(preds_probs, gt_label)

        result = {}

        result['prec1'] = prec1
        result['prec5'] = prec5
        return result


class SkeletonActionMAEEvaluator(SkeletonActionEvaluator):
    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):
        super().__init__(dataset_name, config, distributed, output_dir)

    def process(self, inputs, outputs):
        # TODO: choose using a more readable way
        try:
            gt_label = inputs['label'][:,inputs['attr_begin'][0]:inputs['attr_end'][0]]
        except:
            gt_label = inputs['label']

        # import pdb;pdb.set_trace()
        if len(gt_label.shape)>1:
            #  change the one-hot label into the index
            gt_label = torch.argmax(gt_label, dim=1)
        #  see in the model entry, the output of the label branch is a value of the key "pred"
        try:
            preds_probs = outputs['pred']['logit'][:,inputs['attr_begin'][0]:inputs['attr_end'][0]]
        except:
            preds_probs = outputs['pred']['logit']
            
        self.gt_label.append(gt_label)
        self.preds_probs.append(preds_probs)

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
