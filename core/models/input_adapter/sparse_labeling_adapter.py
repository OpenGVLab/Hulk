# Base on M-MAE
# https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/input_adapters.py
import os
import math
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
from transformers import BertTokenizer, BertModel

import einops
try:
    import xformers
    import xformers.ops
    ATTENTION_MODE = 'xformers'
except:
    ATTENTION_MODE = 'math'

__all__ = ['SpatialTemporalSeq2DInputAdapter', 'SpatialTemporalSeq2DSkActionInputAdapter']

# A placeholder adapter in Hulk, act as the label adapter branch for the detection task.
# The label adapter branch only outputs the token length of the prediction targets.
# The output token from the label adapter branch will be 100% masked in the backbone module for supervised learning.
def sparse_labeling_adapter(pretrained=False, load_pos_embed=True, **kwargs):
    default = dict(
        num_joints=25, num_frames=196, use_abs_pos_emb=True,
    )
    recursive_update(default, kwargs)
    adapter = SpatialTemporalSeq2DInputAdapter(**default)

    if pretrained:
        script_dir = os.path.dirname(__file__)
        script_dir = script_dir.replace('input_adapter', 'backbones')

        if pretrained == 'supervised-80ecf9dd':
            rel_path = "pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
        elif pretrained == 'clip':
            rel_path = "pretrain_weights/CLIP-ViT-B-16.pt"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
            # rename & clean loaded keys
            checkpoint = clip_checkpoint_preprocess(checkpoint)
        elif pretrained == 'HAP':
            rel_path = "pretrain_weights/HAP.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        elif pretrained == 'humanbench':
            rel_path = "pretrain_weights/humanbench.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        else:
            rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']

        # load while interpolates position embedding
        load_checkpoint_adpater(adapter, checkpoint, load_pos_embed, strict=False, logger=dummy_logger, proj_name=adapter.proj_name)
        del checkpoint

    return adapter


# sparse labeling adapter for skeleton action
def sparse_labeling_adapter_skaction(pretrained=False, load_pos_embed=True, **kwargs):
    default = dict(
        num_joints=25, num_frames=196, use_abs_pos_emb=True,
    )
    recursive_update(default, kwargs)
    adapter = SpatialTemporalSeq2DSkActionInputAdapter(**default)

    if pretrained:
        script_dir = os.path.dirname(__file__)
        script_dir = script_dir.replace('input_adapter', 'backbones')

        if pretrained == 'supervised-80ecf9dd':
            rel_path = "pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
        elif pretrained == 'clip':
            rel_path = "pretrain_weights/CLIP-ViT-B-16.pt"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))
            # rename & clean loaded keys
            checkpoint = clip_checkpoint_preprocess(checkpoint)
        elif pretrained == 'HAP':
            rel_path = "pretrain_weights/HAP.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        elif pretrained == 'humanbench':
            rel_path = "pretrain_weights/humanbench.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']
        else:
            rel_path = "pretrain_weights/mae_pretrain_vit_base.pth"
            checkpoint = torch.load(os.path.join(script_dir, rel_path))['model']

        # load while interpolates position embedding
        load_checkpoint_adpater(adapter, checkpoint, load_pos_embed, strict=False, logger=dummy_logger, proj_name=adapter.proj_name)
        del checkpoint

    return adapter


ntu_body_joints = [
    'base of spine',
    'middle of spine',
    'neck',
    'head',
    'left shoulder',
    'left elbow',
    'left wrist',
    'left hand',
    'right shoulder',
    'right elbow',
    'right wrist',
    'right hand',
    'left hip',
    'left knee',
    'left ankle',
    'left foot',
    'right hip',
    'right knee',
    'right ankle',
    'right foot',
    'spine',
    'tip of left hand',
    'left thumb',
    'tip of right hand',
    'right thumb'
]


class SpatialTemporalSeq2DSkActionInputAdapter(nn.Module):
    """Adapter for 2D spatial temporal sequential inputs, like keypoints sequences
    Create tokens from spatial temporal cubes over keypoints.
    :param in_chans (int): Number of input channels.
    :param num_joints (int): Number of joints in each frame.
    :param num_joints (int): Number of frames in each video.
    :param stride_level (tuple): Spatial and temporal stride number. Eg. stride=(1,2)
            for real stride (patch_size[0], patch_size[1]//2)
    :param embed_dim (int): Dimension of output tokens. Can be set using init method.
    :param patch_size (tuple or list): Shape of patch tokens.
    :param learnable_pos: Set to True to learn positional embeddings instead.
    :param task_sp_list (tuple): Set by task-specific parameters.
    :param type_embed (bool): Set by whether adding a type embedding.
    :param pretrained (bool): Set by whether using pretrained weights.
    :param proj_norm (str): Set by the normalization layer in the projection head.
    :param LeakyReLUParam (float): Set by the negative slope of LeakyReLU.
    :param joint_to_positional_embedding_preprocess (bool): Set by whether to preprocess the joints into
            3D+1D positional embedding.
    :param joint_with_text_embedding (bool): Set by whether to add text embedding to the joints.
    :param joint_names (list): Set by the names of joints.
    """
    def __init__(self,
                in_chans: int,
                num_joints: int,
                num_frames: int,
                embed_dim: int,
                patch_size: Union[int, Tuple[int, int]],
                stride_level: Union[int, Tuple[int, int]]=(1, 1),
                use_abs_pos_emb: bool = True,
                pos_emb: str = 'abs_pos_emb',
                learnable_pos: bool = False,
                test_pos_mode: Union[bool, str] = False,
                task_sp_list: tuple = (),
                modality_share_list: tuple = (),
                type_embed: bool = True,
                pretrained: bool = False,
                proj_norm: str = 'BN',
                LeakyReLUParam: int = 0.1,
                joint_to_positional_embedding_preprocess: bool = False,
                joint_with_text_embedding: bool = False,
                joint_names: List[str] = 'ntu_body_joints',
                num_vertices_feats: int = 256,
                PE_mean: float = 0,
                PE_std: float = 1,
                input_PE_LN: bool = False,
                stride_text_embedding: bool = False,
                load_mae_proj_weight: bool = False,
                is_2d_dataset = False,
                num_attn_heads = 0,
                proj_fun = None,
                full_padding = False,
                ):

        super().__init__()
        self.in_chans = in_chans
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride_level = stride_level
        self.pos_emb = pos_emb
        self.learnable_pos = learnable_pos
        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list
        self.type_embed = type_embed
        self.pretrained = pretrained
        self.test_pos_mode = test_pos_mode
        self.joint_to_positional_embedding_preprocess = joint_to_positional_embedding_preprocess
        self.joint_with_text_embedding = joint_with_text_embedding
        self.stride_text_embedding = stride_text_embedding
        self.num_vertices_feats = num_vertices_feats
        if self.joint_to_positional_embedding_preprocess:
            #  change the in_chans to 3 * num_vertices_feats, as 3d positional embedding
            self.in_chans = 3 * self.num_vertices_feats
        self.PE_mean = PE_mean
        self.PE_std = PE_std
        self.input_PE_LN = input_PE_LN
        self.is_2d_dataset = is_2d_dataset
        self.proj_fun = proj_fun
        self.full_padding = full_padding

        if self.joint_with_text_embedding:
            text_embedding = torch.load(f'./{joint_names}.pth')
            self.text_embedding = nn.Parameter(torch.zeros_like(text_embedding), requires_grad=False)
            self.text_embedding.data.copy_(text_embedding)   # num_classes, embed_dim

        # real stride when patching
        self.P_T = max(1, self.patch_size[0] // stride_level[0]) # could be dynamic
        self.P_J = max(1, self.patch_size[1] // stride_level[1]) # could be dynamic


        if full_padding is False:
            self.patch_shape = ((num_frames - self.patch_size[0]) // self.P_T + 1,
                    (num_joints- self.patch_size[1]) // self.P_J +1 )  # could be dynamic
            self.num_patches = ((num_frames - self.patch_size[0]) // self.P_T + 1) * \
                            ((num_joints- self.patch_size[1]) // self.P_J +1) # could be dynamic

            if self.stride_text_embedding:
                text_input_length = self.num_joints
                stride_text = self.P_J
                patch_text = self.patch_size[1]
                text_stride_index_lists = []
                for index in range(0 , text_input_length-patch_text+1, stride_text):
                    text_stride_index_lists.append(list(range(index,index+patch_text)))

                #  find the indexes of the text embedding to be merged with a F.conv2d
                self.text_stride_index_lists = torch.tensor(text_stride_index_lists)
                #  conv weight for [out, in, kernel_j, kernel_feats]
                self.merge_kernel_weight = nn.Parameter(torch.ones((1,1,patch_text,1)), requires_grad=True)
                self.merge_kernel_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
                #  batch_text_feats: [num_patches(tokens), num text in one token, embed_dim]
                self.batched_text_feats = torch.index_select(self.text_embedding, dim=0,
                                                            index=self.text_stride_index_lists.flatten()).view(self.patch_shape[1], patch_text, self.text_embedding.shape[1]).cuda()
        else:
            padding_shape = (self.patch_size[0] // 2, self.patch_size[1] // 2)

            self.patch_shape = ((num_frames - self.patch_size[0] + padding_shape[0] * 2) // self.P_T + 1,
                                (num_joints- self.patch_size[1] + padding_shape[1] * 2) // self.P_J + 1 )  # could be dynamic
            self.num_patches = ((num_frames - self.patch_size[0] + padding_shape[0] * 2) // self.P_T + 1) * \
                            ((num_joints- self.patch_size[1] + padding_shape[1] * 2) // self.P_J + 1) # could be dynamic

            self.merge_kernel_weight = nn.Parameter(torch.ones((1, 1, self.patch_size[1], 1)), requires_grad=True)
            self.merge_kernel_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.proj_name = 'proj'
        self.load_mae_proj_weight = load_mae_proj_weight

        if isinstance(LeakyReLUParam, float):
            activation = nn.LeakyReLU(LeakyReLUParam)
        elif LeakyReLUParam == 'RReLU':
            activation = nn.RReLU()
        elif LeakyReLUParam == 'ReLU':
            activation = nn.ReLU()
        else:
            raise ValueError("LeakyReLUParam must be float, RReLU or ReLU!")

        if proj_norm == 'LN':
            self.proj = nn.Sequential(
                LayerNorm2d(self.embed_dim),
                activation)
            if self.load_mae_proj_weight:
                self.proj_name = 'proj.0'

        self.proj_kernel_weight = nn.Parameter(torch.ones((self.embed_dim, 4, self.patch_size[0], self.patch_size[1])), requires_grad=True)
        self.proj_kernel_bias = nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)

        if self.input_PE_LN:
            self.PE_LN = nn.LayerNorm(self.embed_dim)
        else:
            self.PE_LN = None

        # freeze positional embedding for reconstruction
        if pos_emb == 'abs_pos_emb' or use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_st_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_shape, cls_token=False)
            self.pos_embed.data.copy_(pos_embed.float())
        elif pos_emb.startswith('fixed_pos_emb_'):
            t_shape = pos_emb.replace('fixed_pos_emb_', '')
            t_shape = int(t_shape)
            self.t_shape = t_shape
            pos_emb_res = t_shape * self.patch_shape[1]
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_emb_res, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_st_sincos_pos_embed(self.pos_embed.shape[-1], (t_shape, self.patch_shape[1]), cls_token=False)
            self.pos_embed.data.copy_(pos_embed.float())
        else:
            raise ValueError("Currently we do Not support relative positional embed")

        self.type_embed = nn.Embedding(1, embed_dim) if type_embed else None

        self.num_attn_heads = num_attn_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3) if num_attn_heads > 0 else None

        self.attn_mask = torch.ones((17,17),dtype = torch.bool)

    @staticmethod
    def _normalization(x, PE_mean, PE_std):
        assert len(x.shape) == 5
        # TODO: check normalization function
        if PE_mean!=0 and PE_std!=1:
            x = (x - PE_mean) / PE_std
        return x

    def forward_proj(self, x, shape=None, **kwargs):
        B = x.shape[0] #(N * M, C, T, V)

        if self.is_2d_dataset:
            # proj_kernel_weight = self.proj_kernel_weight[:,:2,:,:]
            proj_kernel_weight = self.proj_kernel_weight[:,[0,1,3],:,:] # xyz,y is depth, y is height
        else:
            proj_kernel_weight = self.proj_kernel_weight[:,[0,1,2],:,:] # xyzp, x,y,z, no probability

        if self.full_padding is False:
            x = F.conv2d(x, weight=proj_kernel_weight,bias=self.proj_kernel_bias,stride=(self.P_T, self.P_J))  # [11, 1, 1, 768]
        else:
            x = F.conv2d(x, weight=proj_kernel_weight,bias=self.proj_kernel_bias,stride=(self.P_T, self.P_J), padding=(self.patch_size[0]//2, self.patch_size[1]//2))  # [11, 1, 1, 768]

        x = self.proj(x)  #->N*M,embed_dim,Tp,Jp


        Tp, Jp = x.shape[2], x.shape[3]

        if self.joint_with_text_embedding and self.stride_text_embedding:
            x = self.forward_Bert_feats_addition(x)
        if self.num_attn_heads > 0:
            # math attn
            x = x.permute(0,2,3,1) #nm,tp,jp,dim
            qkv = self.qkv_proj(x) #nm,tp,jp,dim*3
            qkv = einops.rearrange(qkv, 'B T J (K H D) -> K B (T J) H D', K=3, H=self.num_attn_heads).permute(0,1,3,2,4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H T*J D
            attn = ((q * self.scale) @ k.transpose(-2, -1))
            attn_mask = self.attn_mask.repeat(Tp, Tp).cuda()  # 459,459
            attn = attn.masked_fill(attn_mask == False, float('-inf'))
            attn = attn.softmax(dim=-1) # B H T*J D
            x = (attn @ v).transpose(1, 2) # B T*J H D
            x = x.reshape(B, Tp, Jp, -1).permute(0,3,1,2) #->N*M,embed_dim,Tp,Jp


        x = x.flatten(2).transpose(1, 2) #nm.tj,dim

        return x, (Tp, Jp), B

    def forward_PE(self, x, Tp, Jp, return_pos_embed=False):
        """
        x (Tensor): input feature
        Tp (int): the size of Temporal dimension of PE
        Jp (int): the size of Spatial dimension of PE
        """
        if self.test_pos_mode is False:
            if x.size(1) == self.pos_emb.size(1):
                x = x + self.pos_embed
            else: # take top-left if pos_embed > x's dimension
                x = x + self.pos_embed.reshape(1, self.patch_shape[0],
                                                self.patch_shape[1].
                                                self.pos_emb.size(2))[:, :Tp, :Jp, :].reshape(1, x.size(1),
                                                                                            self.pos_embed.size(2))
        elif self.test_pos_mode == 'learnable_interpolate':
            patch_shape = (Tp, Jp)
            orig_size = self.patch_shape
            # as in orignal scale
            pos_embed = self.pos_embed

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=patch_shape, mode='bicubic',
                                                        align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed
        else:
            raise NotImplementedError

        if return_pos_embed:
            return x, pos_embed
        else:
            return x, None

    def with_type_embed(self, x):
        batch_size, seq_len, _ = x.size()
        return x + self.type_embed.weight.unsqueeze(0).expand(batch_size, seq_len, 1)

    def forward_Bert_feats_addition(self, x):

        if self.full_padding:
            merge_Bert_feats = F.conv2d(self.text_embedding.unsqueeze(0).unsqueeze(0), weight=self.merge_kernel_weight, bias=self.merge_kernel_bias,
                                                        padding=(self.patch_size[1]//2, 0), stride=(self.P_J, 1))  # [11, 1, 1, 768]
            B = x.shape[0]
            merge_Bert_feats = merge_Bert_feats.permute(0, 3, 1, 2).repeat(B, 1, x.shape[2], 1)
        else:
            merge_Bert_feats = F.conv2d(self.batched_text_feats.unsqueeze(1), weight=self.merge_kernel_weight,
                                        bias=self.merge_kernel_bias)  # [11, 1, 1, 768]
            B = x.shape[0]
            merge_Bert_feats = merge_Bert_feats.permute(1, 3, 2, 0).repeat(B, 1, x.shape[2], 1)
        x = x + merge_Bert_feats
        return x


    def forward(self, input_var):
        output = {}

        x = input_var['sparse_labeling']

        if len(x.shape) == 5:
            pass
        elif len(x.shape) == 6: # more dim for num of clip in skeleton-action task
            x = x.permute(0, 2, 1, 3, 4, 5) # N, M, nc, T, V, C
            N, nc, M, T, V, C = x.shape
            x = x.reshape(N,M*nc,T,V,C)
            x = x.permute(0,4,2,3,1) # N,C,T,V,M*nc

        x = self._normalization(x, self.PE_mean, self.PE_std)
        if self.PE_LN is not None: #none
            x = x.permute(0, 2, 3, 4, 1).contiguous() # N, T, V, M, C
            x = self.PE_LN(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous() # N, C, T, V, M

        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V) # N*M, C, T, V

        output["sparse_labeling"] = x
        output["nested_mask"] = None

        x, (Tp, Jp), B = self.forward_proj(x, shape=(N, C, T, V, M))

        x, pos_embed = self.forward_PE(x, Tp, Jp, return_pos_embed=True)

        output['adapter_output_sparse_labeling'] = {'tokens': x,
                                         'Bs': B,
                                         'N_H': Tp,
                                         'N_W': Jp,
                                         'attn_mask':None,
                                         'pos_embed':pos_embed.repeat(B, 1, 1)} #[x, (Tp, Jp), None, B]
        output['batch_size'] = N
        input_var.update(output)
        del x
        torch.cuda.empty_cache()
        return input_var


# A placeholder adapter in Hulk, act as the label adapter branch for the detection task.
# The label adapter branch only outputs the token length of the prediction targets.
# The output token from the label adapter branch will be 100% masked in the backbone module for supervised learning.
class SpatialTemporalSeq2DInputAdapter(nn.Module):
    """Adapter for 2D spatial temporal sequential inputs, like keypoints sequences
    Create tokens from spatial temporal cubes over keypoints.
    :param in_chans (int): Number of input channels.
    :param num_joints (int): Number of joints in each frame.
    :param num_joints (int): Number of frames in each video.
    :param stride_level (tuple): Spatial and temporal stride number. Eg. stride=(1,2)
            for real stride (patch_size[0], patch_size[1]//2)
    :param embed_dim (int): Dimension of output tokens. Can be set using init method.
    :param patch_size (tuple or list): Shape of patch tokens.
    :param learnable_pos: Set to True to learn positional embeddings instead.
    :param task_sp_list (tuple): Set by task-specific parameters.
    :param type_embed (bool): Set by whether adding a type embedding.
    :param pretrained (bool): Set by whether using pretrained weights.
    :param proj_norm (str): Set by the normalization layer in the projection head.
    :param LeakyReLUParam (float): Set by the negative slope of LeakyReLU.
    :param joint_to_positional_embedding_preprocess (bool): Set by whether to preprocess the joints into
            3D+1D positional embedding.
    :param joint_with_text_embedding (bool): Set by whether to add text embedding to the joints.
    :param joint_names (list): Set by the names of joints.
    """
    def __init__(self,
                in_chans: int,
                num_joints: int,
                num_frames: int,
                embed_dim: int,
                patch_size: Union[int, Tuple[int, int]],
                stride_level: Union[int, Tuple[int, int]]=(1, 1),
                use_abs_pos_emb: bool = True,
                learnable_pos: bool = False,
                test_pos_mode: Union[bool, str] = False,
                task_sp_list: tuple = (),
                modality_share_list: tuple = (),
                type_embed: bool = True,
                pretrained: bool = False,
                proj_norm: str = 'BN',
                LeakyReLUParam: int = 0.1,
                joint_to_positional_embedding_preprocess: bool = False,
                joint_with_text_embedding: bool = False,
                joint_names: List[str] = 'ntu_body_joints',
                num_vertices_feats: int = 256,
                pre_extracted: bool = False,
                PE_mean: float = 0,
                PE_std: float = 1,
                input_PE_LN: bool = False,
                stride_text_embedding: bool = False,
                load_mae_proj_weight: bool = False,
                is_2d_dataset = False,
                ):

        super().__init__()
        self.in_chans = in_chans
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride_level = stride_level
        self.use_abs_pos_emb = use_abs_pos_emb
        self.learnable_pos = learnable_pos
        self.task_sp_list = task_sp_list
        self.modality_share_list = modality_share_list
        self.type_embed = type_embed
        self.pretrained = pretrained
        self.test_pos_mode = test_pos_mode
        self.joint_to_positional_embedding_preprocess = joint_to_positional_embedding_preprocess
        self.joint_with_text_embedding = joint_with_text_embedding
        self.stride_text_embedding = stride_text_embedding
        self.num_vertices_feats = num_vertices_feats
        if self.joint_to_positional_embedding_preprocess:
            #  change the in_chans to 3 * num_vertices_feats, as 3d positional embedding
            self.in_chans = 3 * self.num_vertices_feats
        self.PE_mean = PE_mean
        self.PE_std = PE_std
        self.input_PE_LN = input_PE_LN
        self.is_2d_dataset = is_2d_dataset

        # assert pretrained is False, "TODO: Support Pretrained"

        if self.joint_with_text_embedding:
            if pre_extracted:
                text_embedding = torch.load(f'./{joint_names}.pth')
            self.text_embedding = nn.Parameter(torch.zeros_like(text_embedding), requires_grad=False)
            self.text_embedding.data.copy_(text_embedding)   # num_classes, embed_dim

        # real stride when patching
        self.P_T = max(1, self.patch_size[0] // stride_level[0]) # could be dynamic
        self.P_J = max(1, self.patch_size[1] // stride_level[1]) # could be dynamic

        self.patch_shape = ((num_frames - self.patch_size[0]) // self.P_T + 1,
                            (num_joints- self.patch_size[1]) // self.P_J +1 )  # could be dynamic
        self.num_patches = ((num_frames - self.patch_size[0]) // self.P_T + 1) * \
                           ((num_joints- self.patch_size[1]) // self.P_J +1) # could be dynamic

        if self.stride_text_embedding:
            text_input_length = self.num_joints
            stride_text = self.P_J
            patch_text = self.patch_size[1]
            text_stride_index_lists = []
            for index in range(0 , text_input_length-patch_text+1, stride_text):
                text_stride_index_lists.append(list(range(index,index+patch_text)))

            #  find the indexes of the text embedding to be merged with a F.conv2d
            self.text_stride_index_lists = torch.tensor(text_stride_index_lists)
            #  conv weight for [out, in, kernel_j, kernel_feats]
            self.merge_kernel_weight = nn.Parameter(torch.ones((1,1,patch_text,1)), requires_grad=True)
            self.merge_kernel_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
            #  batch_text_feats: [num_patches(tokens), num text in one token, embed_dim]
            self.batched_text_feats = torch.index_select(self.text_embedding, dim=0,
                                                         index=self.text_stride_index_lists.flatten()).view(self.patch_shape[1], patch_text, self.text_embedding.shape[1]).cuda()

        self.proj_name = 'proj'
        self.load_mae_proj_weight = load_mae_proj_weight

        self.proj_kernel_weight = nn.Parameter(
            torch.ones((self.embed_dim, 3, self.patch_size[0], self.patch_size[1])), requires_grad=True)
        self.proj_kernel_bias = nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)


        if proj_norm == 'LN':
            self.proj = nn.Sequential(
                LayerNorm2d(self.embed_dim),
                nn.LeakyReLU(LeakyReLUParam))
            if self.load_mae_proj_weight:
                self.proj_name = 'proj.0'
        else:
            raise NotImplementedError("proj_norm: {} Not Defined".format(proj_norm))

        if self.input_PE_LN:
            self.PE_LN = nn.LayerNorm(self.embed_dim)
        else:
            self.PE_LN = None

        # freeze positional embedding for reconstruction
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_st_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_shape, cls_token=False)
            self.pos_embed.data.copy_(pos_embed.float())
        else:
            raise ValueError("Currently we do Not support relative positional embed")

        self.type_embed = nn.Embedding(1, embed_dim) if type_embed else None

    @staticmethod
    def _normalization(x, PE_mean, PE_std):
        assert len(x.shape) == 5
        # TODO: check normalization function
        if PE_mean!=0 and PE_std!=1:
            x = (x - PE_mean) / PE_std
        return x

    def forward_proj(self, x, **kwargs):
        B = x.shape[0]
        # import pdb;pdb.set_trace()
        if self.is_2d_dataset:
            proj_kernel_weight = self.proj_kernel_weight[:,[0,2],:,:] # xyz,y is depth, y is height
        else:
            proj_kernel_weight = self.proj_kernel_weight

        x = F.conv2d(x, weight=proj_kernel_weight,bias=self.proj_kernel_bias,stride=(self.P_T, self.P_J))  # [11, 1, 1, 768]

        x = self.proj(x)

        Tp, Jp = x.shape[2], x.shape[3]
        if self.joint_with_text_embedding and self.stride_text_embedding:
            x = self.forward_Bert_feats_addition(x)

        x = x.flatten(2).transpose(1, 2)

        return x, (Tp, Jp), B

    def forward_PE(self, x, Tp, Jp):
        """
        x (Tensor): input feature
        Tp (int): the size of Temporal dimension of PE
        Jp (int): the size of Spatial dimension of PE
        """
        if self.test_pos_mode is False:
            if x.size(1) == self.pos_emb.size(1):
                x = x + self.pos_embed
            else: # take top-left if pos_embed > x's dimension
                x = x + self.pos_embed.reshape(1, self.patch_shape[0],
                                                self.patch_shape[1].
                                                self.pos_emb.size(2))[:, :Tp, :Jp, :].reshape(1, x.size(1),
                                                                                            self.pos_embed.size(2))
        elif self.test_pos_mode == 'learnable_interpolate':
            patch_shape = (Tp, Jp)
            orig_size = self.patch_shape
            # as in orignal scale
            pos_embed = self.pos_embed

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=patch_shape, mode='bicubic',
                                                        align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed
        else:
            raise NotImplementedError

        return x

    def with_type_embed(self, x):
        batch_size, seq_len, _ = x.size()
        return x + self.type_embed.weight.unsqueeze(0).expand(batch_size, seq_len, 1)

    def forward_Bert_feats_addition(self, x):
        merge_Bert_feats = F.conv2d(self.batched_text_feats.unsqueeze(1), weight=self.merge_kernel_weight,
                                    bias=self.merge_kernel_bias)  # [11, 1, 1, 768]
        B = x.shape[0]
        merge_Bert_feats = merge_Bert_feats.permute(1, 3, 2, 0).repeat(B, 1, x.shape[2], 1)
        x = x + merge_Bert_feats
        return x


    def forward(self, input_var):

        output = {}

        x = input_var['sparse_labeling']
        if len(x.shape) == 5:
            pass
        elif len(x.shape) == 6: # more dim for num of clip in skeleton-action task
            N, nc, M, T, V, C = x.shape

            x = x.reshape(-1,M,T,V,C)
            x = x.permute(0,4,2,3,1) # N,C,T,V,M

        x = self._normalization(x, self.PE_mean, self.PE_std)
        if self.PE_LN is not None: #none
            x = x.permute(0, 2, 3, 4, 1).contiguous() # N, T, V, M, C
            x = self.PE_LN(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous() # N, C, T, V, M
        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        output["sparse_labeling"] = x
        output["nested_mask"] = None

        x, (Tp, Jp), B = self.forward_proj(x)

        x = self.forward_PE(x, Tp, Jp)

        output['adapter_output_sparse_labeling'] = {'tokens': x,
                                         'Bs': B,
                                         'N_H': Tp,
                                         'N_W': Jp,
                                         'attn_mask':None} #[x, (Tp, Jp), None, B]
        output['batch_size'] = N

        input_var.update(output)
        del x
        torch.cuda.empty_cache()
        return input_var


def interpolate_pos_embed(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    # class_token and dist_token are kept unchanged
    print(f"[rank {dist.get_rank()}] Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='bicubic', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (b, h*w, c)
    return new_pos_embed


def get_2d_st_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    This function calculates 2D spatial temporal positional embedding in sincos
    manner.
    Currently, we do NOT support cls_token=True
    :param embed_dim (int): dimension of positional embed.
    :param grid_size (tuple): (num_T, num_J)
    :return: [grid_size[0]*grid_size[1], embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert cls_token is False, "Currently, we do NOT support cls_token=True"
    pos_list = list()
    grid_T = grid_size[0]
    grid_J = grid_size[1]
    for tk in range(grid_T):
        for st in range(grid_J):
            pos_list.append(st)

    position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
    pe = torch.zeros(grid_T * grid_J, embed_dim)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.view(grid_size[0] * grid_size[1], embed_dim).unsqueeze(0)
    return pe


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank = dist.get_rank()

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    print("finish load PE")

def load_checkpoint_adpater(model, state_dict, load_pos_embed, strict=False, logger=None, proj_name='proj'):
    """
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        proj_name (str): key of the state_dict in the 'adapter projection' module of the model.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if 'pos_embed' in state_dict:
        if load_pos_embed:
            state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
                                                            patch_shape=model.patch_shape,
                                                            num_extra_tokens=1)
        else:
            del state_dict['pos_embed']
            print("checkpoint pos_embed removed")
    state_dict[f'{proj_name}.weight'] = state_dict.pop('patch_embed.proj.weight')
    state_dict[f'{proj_name}.bias'] = state_dict.pop('patch_embed.proj.bias')
    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
    load_state_dict(model, state_dict, strict, logger)

def clip_checkpoint_preprocess(checkpoint):
    for k in list(checkpoint.keys()):
        if k.startswith('visual'):
            if k in ["visual.proj", "visual.class_embedding"]:
                new_k = k
            elif k.startswith('visual.transformer.resblocks'):
                new_k = k[len("visual.transformer.res"):]
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
                new_k = new_k.replace('out_proj', 'proj')
                new_k = new_k.replace('ln_', 'norm')
                new_k = new_k.replace('c_fc', 'fc1')
                new_k = new_k.replace('c_proj', 'fc2')
            else:
                new_k = k[len("visual."):]
                new_k = new_k.replace('positional_embedding', 'pos_embed')
                new_k = new_k.replace('conv1', 'patch_embed.proj')
                new_k = new_k.replace('ln_post', 'norm')
            checkpoint[new_k] = checkpoint[k]
        del checkpoint[k]
    return checkpoint

class dummy_logger:
    def info(self, **kwargs):
        print(**kwargs)

    def warning(self, **kwargs):
        print(**kwargs)


class LayerNorm2d(nn.Module):
    def __init__(self, embed_dim):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.reshape(x.size(0), self.embed_dim, -1)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), self.embed_dim, h, w)
        return x
