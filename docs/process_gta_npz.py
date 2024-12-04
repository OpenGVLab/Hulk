import numpy as np
import pickle
import os
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='OpenXD GTA Human Dataset')
    parser.add_argument('--npz_file', type=str, default='./OpenXD-GTA-Human/gta_human_4x.npz')
    parser.add_argument('--anno_dir', type=str, default='./OpenXD-GTA-Human/gta_human/annotations/')
    parser.add_argument('--output_dir', type=str, default='./OpenXD-GTA-Human/gta_human/dataset_pkl/')

    args = parser.parse_args()
    return args


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img

args = parse_args()


dataset_pkl = {}
image_name_list = []
annotations_list = []

npz_file = args.npz_file
npz_dataset = np.load(npz_file)

# ['__key_strict__', '__temporal_len__', '__keypoints_compressed__', 'config', 'smpl', 'keypoints2d', 'keypoints2d_mask', 'keypoints3d', 'keypoints3d_mask', 'keypoints2d_gta', 'keypoints2d_gta_mask', 'keypoints3d_gta', 'keypoints3d_gta_mask', 'image_path', 'bbox_xywh']

image_path_all = npz_dataset['image_path']
joint24 = ['R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear']

J100_to_24 = [18, 17, 16, 19, 20, 21, 6, 5, 4, 8, 9, 10, 2, 0, 14, 11, 12, 72, 1, 99, 54, 60, 99, 99]

J24_to_24 = [3, 1, 15, 16, 0, 2, 9, 7, 5, 4, 6, 8, 17, 18, 19, 20, 21, 22, 23, 10, 12, 11, 14, 13]

# ['left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'right_hip_extra', 'left_hip_extra', 'neck_extra', 'headtop', 'pelvis_extra', 'thorax_extra', 'spine_extra', 'jaw_extra', 'head_extra']

# SMPL keypoint convention used by SPIN, EFT and so on
SMPL_24_KEYPOINTS = [
    # 24 Keypoints
    'right_ankle',
    'right_knee',
    'right_hip_extra',  # LSP
    'left_hip_extra',  # LSP
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck_extra',  # LSP
    'headtop',  # LSP mpii peen_action mpi_inf_3dhp
    'pelvis_extra',  # MPII
    'thorax_extra',  # MPII
    'spine_extra',  # H36M
    'jaw_extra',  # H36M
    'head_extra',  # H36M
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear'
]
bbox_xywh_npz_dataset = npz_dataset['bbox_xywh']
keypoints2d_npz_dataset = npz_dataset['keypoints2d']
keypoints3d_npz_dataset = npz_dataset['keypoints3d']
# dict_keys(['is_male', 'fov', 'keypoints_2d', 'keypoints_3d', 'occ', 'self_occ', 'num_frames', 'weather', 'daytime', 'location_tag', 'betas', 'global_orient', 'body_pose', 'transl', 'bbox_xywh'])
anno_dict = args.anno_dir

anno_list = sorted(os.listdir(anno_dict))
total_idx = 0
for anno_file in anno_list:
    anno_file_path = os.path.join(anno_dict, anno_file)
    with open(anno_file_path, 'rb') as f:
        mm = pickle.load(f)
        if mm['is_male']:
            gender = 'm'
        else:
            gender = 'f'
        for idx in range(mm['num_frames']):
            if 'images/' + anno_file[:-4] + '/' + str(idx).zfill(8) + '.jpeg' in image_path_all:
                # bbox_xywh = mm['bbox_xywh'][idx]
                bbox_xywh = bbox_xywh_npz_dataset[total_idx]
                keypoints_2d = mm['keypoints_2d'][idx]
                keypoints_3d = mm['keypoints_3d'][idx]
                betas = mm['betas'][idx]
                global_orient = mm['global_orient'][idx]
                body_pose = mm['body_pose'][idx]
                cam_param = mm['transl'][idx]
                
                annotations = {}
                annotations['center'] = [bbox_xywh[0]+bbox_xywh[2]/2, bbox_xywh[1]+bbox_xywh[3]/2]
                annotations['scale'] = bbox_xywh[2:].max()/200
                annotations['has_2d_joints'] = 1
                annotations['has_3d_joints'] = 1

                annotations['2d_joints'] = keypoints2d_npz_dataset[total_idx][J24_to_24]
                annotations['3d_joints'] = keypoints3d_npz_dataset[total_idx][J24_to_24]
                annotations['has_smpl'] = 1
                annotations['pose'] = np.concatenate((global_orient,body_pose), 0).reshape(-1)
                annotations['betas'] = betas
                annotations['gender'] = gender
                annotations['cam'] = cam_param
                
                img_key = image_path_all[total_idx]
                image_name_list.append(img_key)
                annotations_list.append(annotations)
                total_idx += 1

                if total_idx % 10 == 0:
                    print(total_idx)
                
                if (total_idx % 200000 == 0 and total_idx > 0):
                    dataset_pkl['image_name'] = image_name_list
                    dataset_pkl['annotations'] = annotations_list
                    with open(args.output_dir + str(total_idx) + ".pkl", "wb") as f:
                        pickle.dump(dataset_pkl, f)
                    dataset_pkl = {}
                    image_name_list = []
                    annotations_list = []
    
dataset_pkl['image_name'] = image_name_list
dataset_pkl['annotations'] = annotations_list
with open(args.output_dir + str(total_idx) + ".pkl", "wb") as f:
    pickle.dump(dataset_pkl, f)