# Base on M-MAE
# https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/input_adapters.py
import os
import pdb
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from core.utils import NestedTensor
from dict_recursive_update import recursive_update
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

class TextAdapter(nn.Module):
    """
    Text Adapter for text embedding. This one is now a placeholder, only support bert embedding.
    :param: patch_size: placeholder, achieve similar kwargs with other adapters, which is defined in the solver_mae_devnew.py
    :param: in_chans: placeholder, achieve similar kwargs with other adapters
    :param: stride_level: placeholder, achieve similar kwargs with other adapters
    :param: task_sp_list: List of task specific list for DDP communication. Default: ()
    :param: description_dict_name: the name of the dictionary of all description names in datasets
    :param: one_way_semantics: whether only consider the positive semantics, which is the 1 in the label.
    :param: skeleton_action: whether the label is skeleton action, such as semantics in ntu60.
                            in skeleton action, the label of each sample contains M(2) people,
                            each person has N(25) joints, so the batch size is B*M.
    """

    def __init__(self,
                 patch_size=[None, None],  # placeholder, achieve similar kwargs with other adapters
                 in_chans=None,  # placeholder, achieve similar kwargs with other adapters
                 stride_level=None,  # placeholder, achieve similar kwargs with other adapters
                 task_sp_list=(),
                 modality_share_list=(),
                 embed_dim=768,
                 description_dict_name='rap2_attr_name',
                 one_way_semantics=True,
                 skeleton_action=False,
                 skeleton_action_one_hot_label=False,
                 people_cnt=2,
                 image_caption=False,
                 max_tokens=30,
                 ):
        super(TextAdapter, self).__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.stride_level = stride_level
        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list
        self.one_way_semantics = one_way_semantics
        self.skeleton_action = skeleton_action
        self.skeleton_action_one_hot_label = skeleton_action_one_hot_label
        self.people_cnt = people_cnt
        self.image_caption = image_caption
        self.embed_dim = embed_dim


        if not self.image_caption:
            if isinstance(description_dict_name, list):
                text_vectors = []
                for this_dict_name in description_dict_name:
                    text_vectors.append(torch.load(f'./{this_dict_name}.pth'))
                text_vectors = torch.cat(text_vectors, dim=0)
            else:
                text_vectors = torch.load(f'./{description_dict_name}.pth')
            self.text_vectors = nn.Parameter(torch.zeros_like(text_vectors), requires_grad=False)
            self.text_vectors.data.copy_(text_vectors)
            num_tokens = self.text_vectors.shape[0]
            self.patch_shape = [1, num_tokens]
            if self.skeleton_action and not self.skeleton_action_one_hot_label:
                self.patch_shape = [1, 1]
        elif self.image_caption:
            self.patch_shape = [1, max_tokens]

    def forward(self, input_var):
        """
        :param input_var:
        :return: input_var: adapter_output_text: {'tokens': text_embedding,
                                                    'Bs': B,
                                                    'N_H': 1,
                                                    'N_W': text_embedding.shape[1],
                                                    'attn_mask':None}
        """
        output = {}

        if not self.image_caption:
            label = input_var['label'] # bs, num_cls of attribute  #64,120
            B = label.shape[0]
            if len(label.shape)>1:
                if self.one_way_semantics:
                    #  only consider the positive semantics, which is the 1 in the label,
                    #  generate fake all-one label to extract the text embedding
                    label = torch.ones_like(label, device=label.device)
                    text_embedding = self.text_vectors[range(len(self.text_vectors)), label.long()] # bs, num_cls of attr, bert_dim
                else:
                    text_embedding = self.text_vectors[None,:,:].repeat(B,1,1) #bs,cls-num,dim
                if self.skeleton_action:
                    assert self.skeleton_action_one_hot_label
                    # for skeleton action, the label of each sample contains M(people_cnt) people,
                    text_embedding = text_embedding[:,None,...]
                    text_embedding = text_embedding.repeat(1,self.people_cnt, 1, 1)
                    text_embedding = text_embedding.reshape(B*self.people_cnt, text_embedding.shape[-2], text_embedding.shape[-1])
            else:
                #  label example:  tensor([ 8, 33, 39, 34,  6, 14, 1 ]), which is a single classification label
                assert self.one_way_semantics
                text_embedding = self.text_vectors[:, 1]  # num_cls, bert_dim
                text_embedding = text_embedding[label.long()][:, None, :] # bs, 1, bert_dim
                if self.skeleton_action:
                    text_embedding = text_embedding.repeat(1, 2, 1) # bs, 2, bert_dim
                    text_embedding = text_embedding.reshape(B * 2, 1, text_embedding.shape[-1]) # bs*2, 1, bert_dim
        elif self.image_caption:
            token_id, token_type_id, token_padding_mask = input_var['input_id'], input_var['token_type_id'], input_var['padding_mask']
            if self.training:
                text_embedding = torch.zeros(token_id.shape[0], token_id[:, :-1].shape[1], self.embed_dim).cuda()
            else:
                text_embedding = torch.zeros(token_id.shape[0], token_id.shape[1], self.embed_dim).cuda()

            B = token_id.shape[0]

        if text_embedding.shape[-1]!=self.embed_dim:
            text_embedding = torch.zeros(text_embedding.shape[0], text_embedding.shape[1], self.embed_dim).cuda()
        output['adapter_output_text'] = {'tokens': text_embedding,
                                         'Bs': B if not self.skeleton_action else B*self.people_cnt,
                                         'N_H': 1, # assume the text is a two-dim sequence, which has the shape of [1, num_tokens]
                                         'N_W': text_embedding.shape[1],
                                         'attn_mask':None}

        input_var.update(output)

        return input_var

def extract_bert_features(rap2_attr_name, bert_tokenizer, bert_model, dim=768):
    """
    Extract the bert features for all sentences in rap2_attr_name
    :param rap2_attr_name: the dictionary of all attribute names in rap2
    :param bert_tokenizer: tokenizer of bert
    :param bert_model: bert model
    :return: rap2_attr_vectors: the bert features of all sentences in rap2_attr_name
    """
    # Traverse the dictionary to get all sentences
    sentences = []
    max_val_idx = 0
    for attr_dict in rap2_attr_name.values():
        for attr_desc in attr_dict.values():
            sentences.append(attr_desc)
        max_val_idx = max(max_val_idx, len(attr_dict)-1)
    # Batch code and input all sentences into BERT model
    input_ids = bert_tokenizer(sentences, return_tensors='pt',padding=True,truncation=True)
    import pdb;
    #Input the whole batch and extract features
    with torch.no_grad():
        last_hidden_states = bert_model(**input_ids)
    # Initialize the output matrix
    rap2_attr_vectors = torch.zeros([len(rap2_attr_name), max_val_idx+1, dim])
    # Select the corresponding sentence features from last_hidden_states
    idx = 0
    for attr_idx, attr_dict in rap2_attr_name.items():
        for val_idx in range(max_val_idx+1):
            if val_idx in attr_dict:
                # use the pooler_output as the representation of the sentence
                rap2_attr_vectors[attr_idx,val_idx] = last_hidden_states[1][idx]
                idx += 1
    # pdb.set_trace()
    return rap2_attr_vectors

def text_adapter(pretrained=True, **kwargs):
    """
    Create a text adapter for text embedding
    :param pretrained: no pretrained model for text adapter
    :param kwargs: other parameters
    :return: adapter: text adapter
    """
    default = dict()
    recursive_update(default, kwargs)
    adapter = TextAdapter(**default)

    return adapter
