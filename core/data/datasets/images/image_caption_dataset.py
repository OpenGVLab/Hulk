import os
import re
import json
import random
import torch
import torchvision
import numpy as np
import pandas as pd
import os.path as osp
from PIL import Image
from collections import defaultdict
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms import PILToTensor, ToTensor
from core.data.transforms.caption_transforms import RandomAugment

def pre_caption(caption, max_words=30):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[ :max_words])
            
    return caption

def data_transforms(split_type='train', img_size=384, min_scale=0.5):
    if split_type == 'train':
        data_transforms = transforms.Compose([                        
            transforms.RandomResizedCrop(img_size, scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 5, isPIL=True, 
                        augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            PILToTensor()
            # ToTensor()
            ]) 
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            PILToTensor(),
            # ToTensor()
            ])
    return data_transforms


class CocoCaption(Dataset):
    """
    Implementation of the dataloader for coco_caption.
    Mainly used in the model training and evaluation.
    Params:
        ginfo: group information for Multitask learning.
        coco_root: root path of coco2014 dataset.
        anno_root: annotation path of coco captions.
        bert_dir: path of bert-base-uncased for loading tokenizer.
        max_words: max length of input captions.
        img_size: image size.
        prompt: given prompt to add before captions.
    """
    def __init__(self, ginfo, max_words=30, img_size=384, beam_size=1, prompt='', split_type='train',
                 cuhk_peds=False, cuhk_peds_root=None, cuhk_peds_anno_root=None, cuhk_peds_gt_root=None,
                 joint_train=False, synth_peds_root=None, joint_train_anno_root=None, coco_train=False,
                 coco_root=None, anno_root=None, bert_dir='', mals_root=None, luperson_root=None):
        self.task_name = ginfo.task_name
        self.rank = dist.get_rank()
        self.prompt = prompt

        # plus one for bos token
        self.max_words = max_words + 1
        self.img_size = img_size
        self.split_type = split_type
        self.beam_size = beam_size

        self.transforms = data_transforms(split_type, img_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower=True)
        self.cuhk_peds = cuhk_peds
        self.joint_train = joint_train
        self.coco_train = coco_train
        if joint_train:
            self.annotation = json.load(open(joint_train_anno_root, 'r'))
            self.cuhk_peds_root = cuhk_peds_root
            self.synth_peds_root = synth_peds_root
            self.mals_root = mals_root
            self.luperson_root = luperson_root
            self.coco_gt_file = cuhk_peds_gt_root
        elif cuhk_peds:
            self.annotation = json.load(open(cuhk_peds_anno_root, 'r'))
            self.coco_gt_file = cuhk_peds_gt_root
            self.coco_root = cuhk_peds_root
        elif coco_train:
            self.coco_root = coco_root
            self.coco_gt_file = osp.join(anno_root, 'coco_gt', 'coco_karpathy_' + split_type + '_gt.json')
            self.annotation = json.load(open(osp.join(anno_root, 'coco_karpathy_' + split_type + '.json'), 'r'))


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        sample = self.annotation[index]
        if self.joint_train and self.split_type == 'train':
            if sample['split'] == 'cuhk_peds':
                image_path = osp.join(self.cuhk_peds_root, sample['image'])
            elif sample['split'] == 'mals':
                image_path = osp.join(self.mals_root, sample['image'])
            elif sample['split'] == 'luperson':
                image_path = osp.join(self.luperson_root, sample['image'])
            else:
                image_path = osp.join(self.synth_peds_root, sample['image'])
        else:
            image_path = osp.join(self.coco_root, sample['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        if self.split_type != 'train':
            caption_id = np.zeros(self.max_words - 1, dtype=np.int32)
            token_type_id = np.zeros(self.max_words - 1, dtype=np.int32)
            caption_pad_mask = np.zeros(self.max_words - 1, dtype=np.int32)
            if self.cuhk_peds:
                img_id = sample['image'].split('.')[0]
            else:
                img_id = sample['image'].split('/')[-1].strip('.jpg').split('_')[-1]
            coco_gt_file = self.coco_gt_file
            beam_size = self.beam_size
            return {'image': image, 'input_id': caption_id, 'image_id': int(img_id) if not self.cuhk_peds else img_id, 
                    'coco_gt_file': coco_gt_file, 'beam_size': beam_size,
                    'token_type_id': token_type_id, 'padding_mask': caption_pad_mask}
        caption = self.prompt + pre_caption(sample['caption'], self.max_words)
        caption_encode = self.tokenizer.encode_plus(caption, max_length=self.max_words, pad_to_max_length=True,
                                                    return_attention_mask=True, return_token_type_ids=True,
                                                    truncation=True)
        caption_id, caption_pad_mask, token_type_id = caption_encode['input_ids'], caption_encode['attention_mask'], caption_encode['token_type_ids']
        caption_id = np.array(caption_id)
        token_type_id = np.array(token_type_id)
        caption_pad_mask = np.array(caption_pad_mask)
        # caption_pad_mask = (1 - np.array(caption_pad_mask)).astype(bool)
        caption = [caption]
        output = {'image': image, 'input_id': caption_id, 'token_type_id': token_type_id, 'padding_mask': caption_pad_mask, 'label': caption_id}
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + \
               f'rank: {self.rank} task: {self.task_name} mode:{"training" if self.split_type == "train" else "inference"} ' \
               f'dataset_len:{len(self.annotation)} augmentation: {self.transforms}'


class CocoCaptiondemo(Dataset):
    """
    Implementation of the dataloader for coco_caption.
    Mainly used in the model training and evaluation.
    Params:
        ginfo: group information for Multitask learning.
        coco_root: root path of coco2014 dataset.
        anno_root: annotation path of coco captions.
        bert_dir: path of bert-base-uncased for loading tokenizer.
        max_words: max length of input captions.
        img_size: image size.
        prompt: given prompt to add before captions.
    """

    def __init__(self, ginfo, max_words=30, img_size=384, beam_size=1, prompt='', split_type='train', demo_dir='/mnt/cache/tangshixiang/wyz_proj/demo_video_unihcpv2/folder0',
                 cuhk_peds=False, cuhk_peds_root=None, cuhk_peds_anno_root=None, cuhk_peds_gt_root=None,
                 joint_train=False, synth_peds_root=None, joint_train_anno_root=None, coco_train=False,
                 coco_root=None, anno_root=None, bert_dir='', mals_root=None, luperson_root=None):
        self.task_name = ginfo.task_name
        self.rank = dist.get_rank()
        self.prompt = prompt

        # plus one for bos token
        self.max_words = max_words + 1
        self.img_size = img_size
        self.split_type = split_type
        self.beam_size = beam_size

        self.transforms = data_transforms(split_type, img_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower=True)
        self.cuhk_peds = cuhk_peds
        self.joint_train = joint_train
        self.coco_train = coco_train
        if joint_train:
            self.annotation = json.load(open(joint_train_anno_root, 'r'))
            self.cuhk_peds_root = cuhk_peds_root
            self.synth_peds_root = synth_peds_root
            self.mals_root = mals_root
            self.luperson_root = luperson_root
            self.coco_gt_file = cuhk_peds_gt_root
        elif cuhk_peds:
            self.annotation = json.load(open(cuhk_peds_anno_root, 'r'))
            self.coco_gt_file = cuhk_peds_gt_root
            self.coco_root = cuhk_peds_root
        elif coco_train:
            self.coco_root = coco_root
            self.coco_gt_file = osp.join(anno_root, 'coco_gt', 'coco_karpathy_' + split_type + '_gt.json')
            self.annotation = json.load(open(osp.join(anno_root, 'coco_karpathy_' + split_type + '.json'), 'r'))
        self.demo_dir = demo_dir


    def __len__(self):
        return len(os.listdir(self.demo_dir))

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        sample = self.annotation[index]
        if self.joint_train and self.split_type == 'train':
            if sample['split'] == 'cuhk_peds':
                image_path = osp.join(self.cuhk_peds_root, sample['image'])
            elif sample['split'] == 'mals':
                image_path = osp.join(self.mals_root, sample['image'])
            elif sample['split'] == 'luperson':
                image_path = osp.join(self.luperson_root, sample['image'])
            else:
                image_path = osp.join(self.synth_peds_root, sample['image'])
        else:
            image_path = osp.join(self.coco_root, sample['image'])
        filename = os.path.join(self.demo_dir, f'frame_{index}.jpg')
        image = Image.open(filename).convert('RGB')
        image = self.transforms(image)
        if self.split_type != 'train':
            caption_id = np.zeros(self.max_words - 1, dtype=np.int32)
            token_type_id = np.zeros(self.max_words - 1, dtype=np.int32)
            caption_pad_mask = np.zeros(self.max_words - 1, dtype=np.int32)
            if self.cuhk_peds:
                img_id = sample['image'].split('.')[0]
            else:
                img_id = sample['image'].split('/')[-1].strip('.jpg').split('_')[-1]
            coco_gt_file = self.coco_gt_file
            beam_size = self.beam_size
            return {'image': image, 'input_id': caption_id, 'image_id': filename,
                    'coco_gt_file': coco_gt_file, 'beam_size': beam_size,
                    'token_type_id': token_type_id, 'padding_mask': caption_pad_mask}
        caption = self.prompt + pre_caption(sample['caption'], self.max_words)
        caption_encode = self.tokenizer.encode_plus(caption, max_length=self.max_words, pad_to_max_length=True,
                                                    return_attention_mask=True, return_token_type_ids=True,
                                                    truncation=True)
        caption_id, caption_pad_mask, token_type_id = caption_encode['input_ids'], caption_encode['attention_mask'], \
        caption_encode['token_type_ids']
        caption_id = np.array(caption_id)
        token_type_id = np.array(token_type_id)
        caption_pad_mask = np.array(caption_pad_mask)
        # caption_pad_mask = (1 - np.array(caption_pad_mask)).astype(bool)
        caption = [caption]
        output = {'image': image, 'input_id': filename, 'token_type_id': token_type_id,
                  'padding_mask': caption_pad_mask, 'label': caption_id}
        return output

    def __repr__(self):
        return self.__class__.__name__ + \
            f'rank: {self.rank} task: {self.task_name} mode:{"training" if self.split_type == "train" else "inference"} ' \
            f'dataset_len:{len(self.annotation)} augmentation: {self.transforms}'
