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

class SMPLMAEEvaluator(DatasetEvaluator):

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
        self.gt_3d_joints = []
        self.has_3d_joints = []
        self.has_smpl = []
        self.gt_vertices_fine = []
        self.pred_vertices = []
        self.pred_3d_joints_from_smpl = []

    def process(self, inputs, outputs):

        self.gt_3d_joints.append(inputs['gt_3d_joints'].cpu())
        self.has_3d_joints.append(inputs['has_3d_joints'].cpu())
        self.has_smpl.append(inputs['has_smpl'].cpu())
        self.gt_vertices_fine.append(inputs['gt_3d_vertices_fine'].cpu())

        self.pred_vertices.append(outputs['pred']["pred_3d_vertices_fine"].cpu())
        self.pred_3d_joints_from_smpl.append(outputs['pred']["pred_3d_joints_from_smpl"].cpu())

        # gt_label = inputs['label']
        # preds_probs = outputs['logit']
        # self.gt_label.append(gt_label)
        # self.preds_probs.append(preds_probs)

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
        gt_3d_joints = torch.cat(self.gt_3d_joints, dim=0)
        has_3d_joints = torch.cat(self.has_3d_joints, dim=0)
        has_smpl = torch.cat(self.has_smpl, dim=0)
        gt_vertices_fine = torch.cat(self.gt_vertices_fine, dim=0)
        pred_vertices = torch.cat(self.pred_vertices, dim=0)
        pred_3d_joints_from_smpl = torch.cat(self.pred_3d_joints_from_smpl, dim=0)


        if self._distributed:
            torch.cuda.synchronize()

            gt_3d_joints = self.all_gather(gt_3d_joints)
            has_3d_joints = self.all_gather(has_3d_joints)
            has_smpl = self.all_gather(has_smpl)
            gt_vertices_fine = self.all_gather(gt_vertices_fine)
            pred_vertices = self.all_gather(pred_vertices)
            pred_3d_joints_from_smpl = self.all_gather(pred_3d_joints_from_smpl)


            if dist.get_rank() != 0:
                return
            gt_vertices_fine = [x.cpu() for x in gt_vertices_fine]
            pred_vertices = [x.cpu() for x in pred_vertices]
            gt_3d_joints = torch.cat(gt_3d_joints, dim=0)
            has_3d_joints = torch.cat(has_3d_joints, dim=0)
            has_smpl = torch.cat(has_smpl, dim=0)
            gt_vertices_fine = torch.cat(gt_vertices_fine, dim=0)
            pred_vertices = torch.cat(pred_vertices, dim=0)
            pred_3d_joints_from_smpl = torch.cat(pred_3d_joints_from_smpl, dim=0)
        print('===')
        print(pred_3d_joints_from_smpl.shape)
        print(gt_3d_joints.shape)
        # measure errors
        np.save('pred_3d_joint_from_smpl.npy', pred_3d_joints_from_smpl.numpy())
        np.save('gt_3d_joint.npy',gt_3d_joints.numpy())

        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices_fine, has_smpl)
        error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints, has_3d_joints)
        error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)

        result = {}

        result['@mPVE'] = np.mean(error_vertices) * 1000
        result['@mPJPE'] = np.mean(error_joints) * 1000
        result['@PAmPJPE'] = np.mean(error_joints_pa) * 1000
        return result



def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error
def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat
def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

