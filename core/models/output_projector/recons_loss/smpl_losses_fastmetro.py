import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dict_recursive_update import recursive_update
from core.data.datasets.images.smpl_data_tools._smpl import SMPL, Mesh
import core.data.datasets.images.smpl_data_tools.config_smpl as smpl_cfg
from core.data.datasets.images.smpl_data_tools import _smpl
from pathlib import Path

class EdgeLengthGTLoss(torch.nn.Module):
    """
    Modified from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/nets/loss.py)
    """
    def __init__(self, face, edge, uniform=True):
        super().__init__()
        self.face = face # num_faces X 3
        self.edge = edge # num_edges X 2
        self.uniform = uniform

    def forward(self, pred_vertices, gt_vertices, has_smpl, device='cuda'):
        face = self.face
        edge = self.edge
        coord_out = pred_vertices[has_smpl == 1]
        coord_gt = gt_vertices[has_smpl == 1]
        if len(coord_gt) > 0:
            if self.uniform:
                d1_out = torch.sqrt(torch.sum((coord_out[:,edge[:,0],:] - coord_out[:,edge[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_edges X 1
                d1_gt = torch.sqrt(torch.sum((coord_gt[:,edge[:,0],:] - coord_gt[:,edge[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_edges X 1
                edge_diff = torch.abs(d1_out - d1_gt)
            else:
                d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

                d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

                diff1 = torch.abs(d1_out - d1_gt)
                diff2 = torch.abs(d2_out - d2_gt)
                diff3 = torch.abs(d3_out - d3_gt)
                edge_diff = torch.cat((diff1, diff2, diff3),1)
            loss = edge_diff.mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device)

        return loss


class EdgeLengthSelfLoss(torch.nn.Module):
    def __init__(self, face, edge, uniform=True):
        super().__init__()
        self.face = face # num_faces X 3
        self.edge = edge # num_edges X 2
        self.uniform = uniform

    def forward(self, pred_vertices, has_smpl, device='cuda'):
        face = self.face
        edge = self.edge
        coord_out = pred_vertices[has_smpl == 1]
        if len(coord_out) > 0:
            if self.uniform:
                edge_self_diff = torch.sqrt(torch.sum((coord_out[:,edge[:,0],:] - coord_out[:,edge[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_edges X 1
            else:
                d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                edge_self_diff = torch.cat((d1_out, d2_out, d3_out),1)
            loss = torch.mean(edge_self_diff)
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device)

        return loss


class NormalVectorLoss(torch.nn.Module):
    """
    Modified from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/nets/loss.py)
    """
    def __init__(self, face):
        super().__init__()
        self.face = face # num_faces X 3

    def forward(self, pred_vertices, gt_vertices, has_smpl, device='cuda'):
        face = self.face
        coord_out = pred_vertices[has_smpl == 1]
        coord_gt = gt_vertices[has_smpl == 1]
        if len(coord_gt) > 0:
            v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
            v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
            v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:] # batch_size X num_faces X 3
            v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

            v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
            v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
            normal_gt = torch.cross(v1_gt, v2_gt, dim=2) # batch_size X num_faces X 3
            normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

            cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            loss = torch.cat((cos1, cos2, cos3),1).mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device)

        return loss

class _SMPL_LOSS_FASTMETRO(nn.Module):
    def __init__(self, configer=None, **kwargs):
        super(_SMPL_LOSS_FASTMETRO, self).__init__()
        self.criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda('cuda')
        self.criterion_keypoints = torch.nn.L1Loss(reduction='none').cuda('cuda')
        self.criterion_vertices = torch.nn.L1Loss().cuda('cuda')

        # define loss functions for edge length & normal vector
        smpl_intermediate_faces = torch.from_numpy(np.load(f'{str(Path(_smpl.__file__).parent)}/smpl_modeling/data/smpl_1723_faces.npy', encoding='latin1', allow_pickle=True).astype(np.int64)).to('cuda')
        smpl_intermediate_edges = torch.from_numpy(np.load(f'{str(Path(_smpl.__file__).parent)}/smpl_modeling/data/smpl_1723_edges.npy', encoding='latin1', allow_pickle=True).astype(np.int64)).to('cuda')
        self.edge_gt_loss = EdgeLengthGTLoss(smpl_intermediate_faces, smpl_intermediate_edges, uniform=True)
        self.edge_self_loss = EdgeLengthSelfLoss(smpl_intermediate_faces, smpl_intermediate_edges, uniform=True)
        self.normal_loss = NormalVectorLoss(smpl_intermediate_faces)

    def keypoint_2d_loss(self, criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
        """
        Compute 2D reprojection loss if 2D keypoint annotations are available.
        The confidence (conf) is binary and indicates whether the keypoints exist or not.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device='cuda'):
        """
        Compute 3D keypoint loss if 3D keypoint annotations are available.
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1].clone()
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(device)

    def vertices_loss(self, criterion_vertices, pred_vertices, gt_vertices, has_smpl, device='cuda'):
        """
        Compute per-vertex loss if vertex annotations are available.
        """
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(device)
        
    def joints_classes_loss(self, pred_logits, gt_joints_label):
        batch_size =  pred_logits.shape[0]
        loss = F.binary_cross_entropy_with_logits(pred_logits, gt_joints_label, reduction='none')
        return loss.sum() / batch_size
        
    def forward(self, outputs, target):

        pred_3d_joints = outputs["pred_3d_joints"]
        pred_3d_vertices_coarse = outputs["pred_3d_vertices_coarse"]
        pred_3d_vertices_intermediate = outputs["pred_3d_vertices_intermediate"]
        pred_3d_vertices_fine = outputs["pred_3d_vertices_fine"]

        pred_3d_joints_from_smpl = outputs["pred_3d_joints_from_smpl"]
        pred_2d_joints = outputs["pred_2d_joints"]
        pred_2d_joints_from_smpl = outputs["pred_2d_joints_from_smpl"]
        pred_logits = outputs["pred_logits"]

        gt_2d_joints = target["gt_2d_joints"]
        has_2d_joints = target["has_2d_joints"]
        gt_3d_joints = target["gt_3d_joints"]
        has_3d_joints = target["has_3d_joints"]
        gt_vertices_fine = target["gt_vertices_fine"]
        gt_vertices_coarse = target["gt_vertices_coarse"]
        gt_vertices_intermediate = target["gt_vertices_intermediate"]
        has_smpl = target["has_smpl"]
        gt_joints_label = target["gt_joints_label"]
        
        # compute 3d joint loss  (where the joints are directly output from transformer)
        loss_3d_joints = self.keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints)

        # compute 3d vertex loss
        loss_vertices = (0.25 * self.vertices_loss(self.criterion_vertices, pred_3d_vertices_coarse, gt_vertices_coarse, has_smpl) + \
            0.5 * self.vertices_loss(self.criterion_vertices, pred_3d_vertices_intermediate, gt_vertices_intermediate, has_smpl) + \
            0.25 * self.vertices_loss(self.criterion_vertices, pred_3d_vertices_fine, gt_vertices_fine, has_smpl))

        # compute 3d joint loss (where the joints are regressed from full mesh)
        loss_reg_3d_joints = self.keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints, has_3d_joints)

        # compute 2d joint loss
        loss_2d_joints = self.keypoint_2d_loss(self.criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints)  + \
                         self.keypoint_2d_loss(self.criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints, has_2d_joints)

        loss_3d_joints = loss_3d_joints + loss_reg_3d_joints
        if self.use_pred_joints_class_loss:
            loss_joints_classes = self.joints_classes_loss(pred_logits, gt_joints_label)
        else:
            loss_joints_classes = 0
        # compute edge length loss (GT supervision & self regularization) & normal vector loss
        loss_edge_normal = (5.0 * self.edge_gt_loss(pred_3d_vertices_intermediate, gt_vertices_intermediate, has_smpl) + \
                            1e-4 * self.edge_self_loss(pred_3d_vertices_intermediate, has_smpl) +\
                            0.1* self.normal_loss(pred_3d_vertices_intermediate, gt_vertices_intermediate, has_smpl))

        # we empirically use hyperparameters to balance difference losses
        # loss = 1000.0*loss_3d_joints + 100.0*loss_vertices + 100.0*loss_edge_normal +100.0*loss_2d_joints

        return {"loss_3d_joints": 1000.0*loss_3d_joints, "loss_vertices": 100.0*loss_vertices, "loss_edge_normal": 100.0*loss_edge_normal, "loss_2d_joints": 100.0*loss_2d_joints, "loss_joints_classes": loss_joints_classes}

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

# Add post processing for smpl loss function
class SMPL_LOSS_FASTMETRO(_SMPL_LOSS_FASTMETRO):
    def __init__(self, configer=None, use_pred_joints_class_loss=False, **kwargs):
        default = dict()
        recursive_update(default, kwargs)
        super(SMPL_LOSS_FASTMETRO, self).__init__(configer, **default)
        self.smpl = SMPL().to('cuda')
        self.mesh_sampler = Mesh()
        self.coarse2intermediate_upsample = nn.Linear(431, 1723)
        self.use_pred_joints_class_loss = use_pred_joints_class_loss



        return

    def forward(self, outputs, raw_targets, mask, modality):
        features = outputs['pred_points']
        x = raw_targets
        # feature.shape = B, C, N_J*N_T, P_J*P_T
        B,C,_,_ = features.shape
        num_joints = 14
        pred_cam =  features[:, :, :,:1].reshape([B,C])# B X 3
        pred_3d_joints_from_token = features[:, :, :,1:1+num_joints].reshape([B,C,-1]).transpose(1,2) # B X 14 X 3
        pred_3d_vertices_coarse = features[:, :, :,1+num_joints:].reshape([B,C,-1]).transpose(1,2) # B X 431 X 3

        # coarse-to-intermediate mesh upsampling

        pred_3d_vertices_intermediate = self.coarse2intermediate_upsample(pred_3d_vertices_coarse.transpose(1,2)).transpose(1,2) # batch_size X num_vertices(intermediate) X 3
        # intermediate-to-fine mesh upsampling
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_intermediate, n1=1, n2=0) # batch_size X num_vertices(fine) X 3
        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_smpl = self.smpl.get_h36m_joints(pred_3d_vertices_fine) # batch_size X 17 X 3
        pred_3d_joints_from_smpl_pelvis = pred_3d_joints_from_smpl[:,smpl_cfg.H36M_J17_NAME.index('Pelvis'),:]
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,smpl_cfg.H36M_J17_TO_J14,:] # batch_size X 14 X 3
        # print('==')
        # print(pred_3d_joints_from_smpl_pelvis.sum(1).mean())

        # normalize predicted vertices
        pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 6890 X 3
        pred_3d_vertices_intermediate = pred_3d_vertices_intermediate - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 1723 X 3
        pred_3d_vertices_coarse = pred_3d_vertices_coarse - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 431 X 3
        # normalize predicted joints
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 14 X 3
        pred_3d_joints_from_token_pelvis = (pred_3d_joints_from_token[:,2,:] + pred_3d_joints_from_token[:,3,:]) / 2
        pred_3d_joints_from_token = pred_3d_joints_from_token - pred_3d_joints_from_token_pelvis[:, None, :] # batch_size X 14 X 3
        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_cam)  # batch_size X 14 X 2
        pred_2d_joints_from_token = orthographic_projection(pred_3d_joints_from_token, pred_cam) # batch_size X 14 X 2
        outputs = {
            "cam_param": pred_cam,
            "pred_3d_joints": pred_3d_joints_from_token,
            "pred_3d_vertices_coarse": pred_3d_vertices_coarse,
            "pred_3d_vertices_intermediate": pred_3d_vertices_intermediate,
            "pred_3d_vertices_fine": pred_3d_vertices_fine,
            "pred_3d_joints_from_smpl": pred_3d_joints_from_smpl,
            "pred_2d_joints_from_smpl": pred_2d_joints_from_smpl,
            "pred_2d_joints": pred_2d_joints_from_token,
            "pred_logits": outputs['pred_logits']
        }



        target = {
            "gt_2d_joints": x['gt_2d_joints'],
            "gt_3d_joints": x['gt_3d_joints'],
            'has_2d_joints': x['has_2d_joints'],
            'has_3d_joints': x['has_3d_joints'],
            'gt_vertices_fine': x['gt_3d_vertices_fine'],
            'gt_vertices_coarse': x['gt_3d_vertices_coarse'],
            'gt_vertices_intermediate' : x['gt_3d_vertices_intermediate'],
            'has_smpl' : x['has_smpl'],
            'gt_joints_label': x['gt_joints_label']
            
        }

        loss_dict = super().forward(outputs, target)
        loss_dict.update(outputs)
        return loss_dict

def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    # import pdb;pdb.set_trace()
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).view(shape)
    return X_2d
