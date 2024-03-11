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
from transformers import BertTokenizer
from pycocotools.coco import COCO
from core.solvers.utils.pycocoevalcap.eval_cuhk import COCOEvalCap

class Image_Caption_Evaluator(DatasetEvaluator):

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
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')

    def reset(self):
        self.all_res = []
        self.coco_file = None

    def process(self, inputs, outputs):

        self.coco_file = inputs['coco_gt_file'][-1]
        # import pdb; pdb.set_trace()

        # tokenizer = inputs['tokenizer']
        image_ids = inputs['image_id']
        all_caps = outputs['pred']
        for img_id, caps in zip(image_ids, all_caps):
            caps = self.tokenizer.decode(caps.tolist(), skip_special_tokens=True)
            # for coco test
            # self.all_res.append({"image_id": img_id.item(), "caption": caps})

            # for cuhk_peds test
            self.all_res.append({"image_id": img_id, "caption": caps})

    @staticmethod
    def all_gather(data, group=0):
        assert dist.get_world_size() == 1, f"distributed eval unsupported yet, uncertain if we can use torch.dist with link jointly"
        if dist.get_world_size() == 1:
            return [data]
        world_size = dist.get_world_size()
        tensors_gather = [torch.ones_like(data) for _ in range(world_size)]
        dist.allgather(tensors_gather, data, group=group)
        return tensors_gather

    def evaluate(self):
        # import pdb; pdb.set_trace()
        result_file = self.save_result(self.all_res, remove_duplicate='image_id')
        result_file = os.path.join(self._output_dir, 'image_caption_test.json')
        # result_file = json.load(open(result_file, 'r'))
        coco_eval = COCOEvalCap(self.coco_file, result_file)
        coco_eval.evaluate()
        result = coco_eval.eval
        # result = {"Generating result file has been saved in:": result_file}
        return result

    def save_result(self, result, remove_duplicate=''):
        result_file = os.path.join(self._output_dir, 'image_caption_test.json')
        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new
        json.dump(result, open(result_file, 'w'))
        print('result file saved to %s' %result_file)
        return result_file
    
    # def coco_caption_eval(self, coco_gt_root, result_file):
    #     coco = COCO(coco_gt_root)
    #     coco_result = coco.loadRes(result_file)
    #     coco_eval = COCOEvalCap(coco, coco_result)
    #     coco_eval.evaluate()
    #     result = coco_eval.eval
    #     return result
