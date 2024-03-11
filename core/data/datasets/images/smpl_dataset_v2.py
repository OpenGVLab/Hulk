"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import cv2
import math
import json
from PIL import Image
import os
import os.path as op
import numpy as np
import code
import scipy.misc
from tqdm import tqdm
import yaml
import errno
import torch
from .smpl_data_tools.tsv_file import load_linelist_file, CompositeTSVFile, \
                                TSVFile, find_file_path_in_yaml, img_from_base64
from .smpl_data_tools.image_ops import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from .smpl_data_tools import config_smpl as smpl_cfg
from .smpl_data_tools._smpl import SMPL, Mesh
import torch.distributed as dist
import pickle
from easydict import EasyDict as edict

class MeshTSVDataset(object):
    def __init__(self, data_file, root_path, is_train=True, cv2_output=False,
                 augmentation={}, ginfo=None):  

        self.data_file = data_file
        self.root_path = root_path
        
        self.image_path_list, self.image_name_list, self.annotation_list = self.get_dataset_info(self.data_file, self.root_path, self.is_composite)

        self.cv2_output = cv2_output
        self.is_train = is_train
        self.scale_factor = augmentation.get('scale_factor', 0.25) # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = augmentation.get('noise_factor', 0.4)
        self.rot_factor = augmentation.get('rot_factor', 30) # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = augmentation.get('img_res', 224)


        self.joints_definition = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
        self.pelvis_index = self.joints_definition.index('Pelvis')
        
        self.len = len(self.image_path_list)

        self.task_name = ginfo.task_name
        self.device = 'cpu' #dist.get_rank() % 8
        self.smpl_init = False

        
    def get_dataset_info(self, data_file, root_path, is_composite=False):
        image_path = []
        image_name_list = []
        annotations = []
        if is_composite:
            for sub_data_file, sub_root_path in zip(data_file, root_path):
                with open(sub_data_file, 'rb') as f:
                    dataset_info = pickle.load(f)

                image_path.extend([os.path.join(sub_root_path, image_name) for image_name in dataset_info['image_name']])
                image_name_list.extend(dataset_info['image_name'])
                annotations.extend(dataset_info['annotations'])
        else:
            with open(data_file, 'rb') as f:
                dataset_info = pickle.load(f)

            image_path.extend([os.path.join(root_path, image_name) for image_name in dataset_info['image_name']])
            image_name_list.extend(dataset_info['image_name'])
            annotations.extend(dataset_info['annotations'])
        return image_path, image_name_list, annotations
    

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling

        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                      [self.img_res, self.img_res], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,255]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def base64tocv2(self, base64):
        cv2_im = img_from_base64(base64)
        if self.cv2_output:
            return cv2_im.astype(np.float32, copy=True)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)


    def __len__(self):
        return self.len

    def _init_smpl(self):
        if not self.smpl_init:
            self.smpl = SMPL().to(self.device)
            self.mesh_sampler = Mesh(device=self.device)
            self.smpl_init = True

    def __getitem__(self, idx):
        self._init_smpl()
        # self._init_tsv()

        img_path = self.image_path_list[idx]
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
        except:
            print('{} load failed'.format(img_path))
        
        img_key = self.image_name_list[idx]
        annotations = self.annotation_list[idx]

        # annotations = annotations[0]
        center = annotations['center']
        scale = annotations['scale']
        has_2d_joints = annotations['has_2d_joints']
        has_3d_joints = annotations['has_3d_joints']
        joints_2d = np.asarray(annotations['2d_joints'])
        joints_3d = np.asarray(annotations['3d_joints'])

        if joints_2d.ndim==3:
            joints_2d = joints_2d[0]
        if joints_3d.ndim==3:
            joints_3d = joints_3d[0]

        # Get SMPL parameters, if available
        has_smpl = np.asarray(annotations['has_smpl'])
        pose = np.asarray(annotations['pose']).reshape(-1)
        betas = np.asarray(annotations['betas'])
        
        if 'vertices' in annotations.keys():
            vertices = np.asarray(annotations['vertices'])
            
        try:
            gender = annotations['gender']
        except KeyError:
            gender = 'none'

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        # Process image
        try:
            img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        except:
            import pdb;pdb.set_trace()

        img = torch.from_numpy(img).float()

        # Store image before normalization to use it in visualization
        # transfromed_img  = self.normalize_img(img)
        transfromed_img = img

        # normalize 3d pose by aligning the pelvis as the root (at origin)
        root_pelvis = joints_3d[self.pelvis_index,:-1]
        joints_3d[:,:-1] = joints_3d[:,:-1] - root_pelvis[None,:]
        # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
        joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
        # 2d pose augmentation
        joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)

        meta_data = {}
        meta_data['ori_img'] = img
        meta_data['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['has_3d_joints'] = has_3d_joints
        meta_data['has_smpl'] = has_smpl
        meta_data['gt_joints_label'] = torch.from_numpy(np.ones(14)).float()

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = has_2d_joints
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
        meta_data['scale'] = float(sc * scale)
        meta_data['center'] = np.asarray(center).astype(np.float32)
        meta_data['gender'] = gender

        meta_data['img_key'] = img_key
        meta_data['image'] = transfromed_img
        if 'vertices' in annotations.keys():
            meta_data['vertices'] = torch.from_numpy(self.pose_processing(vertices, rot, flip)).float()

        smpl_target = self.prepare_smpl_targets(meta_data)
        meta_data.update(smpl_target)

        # output
        output = {}
        output['image'] = transfromed_img

        output.update(meta_data)

        return output

    def prepare_smpl_targets(self, targets):
        new_targets = {}

        # gt 2d joints
        gt_2d_joints= targets['joints_2d'] # 24 x 3
        gt_2d_joints = gt_2d_joints[smpl_cfg.J24_TO_J14, :] # 14 x 3
        has_2d_joints = targets['has_2d_joints'] # 1

        # gt 3d joints
        gt_3d_joints = targets['joints_3d'] # 24 x 4
        gt_3d_pelvis = gt_3d_joints[smpl_cfg.J24_NAME.index('Pelvis'),:3] # 4
        gt_3d_joints = gt_3d_joints[smpl_cfg.J24_TO_J14,:] # 14 X 4
        gt_3d_joints[:,:3] = gt_3d_joints[:,:3] - gt_3d_pelvis[None, :] # 14 X 4
        has_3d_joints = targets['has_3d_joints'] # 1

        # gt params for smpl
        gt_pose = targets['pose'].to(self.device) #  72
    
        
        gt_betas = targets['betas'].to(self.device) #  10
        has_smpl = targets['has_smpl'] # 1

        # generate simplified mesh
        if 'vertices' in targets.keys():
            vertices = targets['vertices'].to(self.device)
            gt_3d_vertices_fine = vertices
        else:
            gt_3d_vertices_fine = self.smpl(gt_pose[None,:], gt_betas[None,:]).squeeze(0) #  6890 X 3
        gt_3d_vertices_intermediate = self.mesh_sampler.downsample(gt_3d_vertices_fine[None,:], n1=0, n2=1).squeeze(0) #  1723 X 3
        gt_3d_vertices_coarse = self.mesh_sampler.downsample(gt_3d_vertices_intermediate[None,:], n1=1, n2=2).squeeze(0) #  431 X 3

        # normalize ground-truth vertices & joints (based on smpl's pelvis)
        # smpl.get_h36m_joints: from vertex to joint (using smpl)
        gt_smpl_3d_joints = self.smpl.get_h36m_joints(gt_3d_vertices_fine[None,:]).squeeze(0) #  17 X 3
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[smpl_cfg.H36M_J17_NAME.index('Pelvis'),:] #  3
        gt_3d_vertices_fine = gt_3d_vertices_fine - gt_smpl_3d_pelvis[ None, :] #  6890 X 3
        gt_3d_vertices_intermediate = gt_3d_vertices_intermediate - gt_smpl_3d_pelvis[ None, :] # 1723 X 3
        gt_3d_vertices_coarse = gt_3d_vertices_coarse - gt_smpl_3d_pelvis[None, :] #  431 X 3

        gt_pose = gt_pose.cpu()
        gt_betas = gt_betas.cpu()
        gt_3d_vertices_fine = gt_3d_vertices_fine.cpu()
        gt_3d_vertices_intermediate = gt_3d_vertices_intermediate.cpu()
        gt_3d_vertices_coarse = gt_3d_vertices_coarse.cpu()

        new_targets['gt_2d_joints'] = gt_2d_joints
        new_targets['has_2d_joints'] = has_2d_joints
        new_targets['gt_3d_joints'] = gt_3d_joints
        new_targets['has_3d_joints'] = has_3d_joints
        new_targets['gt_3d_vertices_coarse'] = gt_3d_vertices_coarse
        new_targets['gt_3d_vertices_intermediate'] = gt_3d_vertices_intermediate
        new_targets['gt_3d_vertices_fine'] = gt_3d_vertices_fine
        new_targets['has_smpl'] = has_smpl
        new_targets['gt_pose'] = gt_pose
        new_targets['gt_betas'] = gt_betas

        camera_placeholder = torch.zeros([1,3]) # 1 X 3
        new_targets['sparse_labeling'] = torch.cat((camera_placeholder, gt_3d_joints[:,:-1], gt_3d_vertices_coarse), 0).permute(1, 0).contiguous()[:,None,:,None]    # C, T, V, M 3, 1, 1+14+431, 1

        return new_targets



class MeshTSVYamlDataset(MeshTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, cfg, is_composite=True, is_train=True, cv2_output=False, augmentation=dict(), ginfo=None):
        self.cfg = cfg
        self.is_composite = is_composite
        data_file = self.cfg.get('data_path', None)
        root_path = self.cfg.get('root_path', None)

        super(MeshTSVYamlDataset, self).__init__(
            data_file, root_path, is_train, cv2_output=cv2_output, augmentation=augmentation, ginfo=ginfo)
