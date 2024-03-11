import random
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import pickle
from torch.nn.modules.utils import _pair
from typing import Dict, List, Optional, Tuple, Union, Sequence

EPS = 1e-4

class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results


def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot3d(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot
#
# def _rot2d(rot):
#     """
#     rot: T,3
#     """
#     cos_r, sin_r = rot.cos(), rot.sin()  # T,2
#     zeros = torch.zeros(rot.shape[0], 1)  # T,1
#     ones = torch.ones(rot.shape[0], 1)  # T,1
#
#     r1 = torch.stack((ones, zeros),dim=-1)  # T,1,2
#     rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,2
#     rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
#     rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3
#
#     ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
#     r2 = torch.stack((zeros, ones),dim=-1)
#     ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
#     ry = torch.cat((ry1, r2, ry3), dim = 1)
#
#     rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
#     r3 = torch.stack((zeros, zeros, ones),dim=-1)
#     rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
#     rz = torch.cat((rz1, rz2, r3), dim = 1)
#
#     # rot = rz.matmul(ry).matmul(rx)
#     rot = ry.matmul(rx)
#     return rot



def _rot2d(rot):
    """
    rot: T, 1 (rotation angles in radians)
    """
    cos_theta = torch.cos(rot)  # Calculate the cosine of the rotation angle
    sin_theta = torch.sin(rot)  # Calculate the sine of the rotation angle

    # Construct the 2D rotation matrix
    rz1 = torch.stack((cos_theta, -sin_theta), dim=-1)  # Shape: T, 1, 2
    rz2 = torch.stack((sin_theta, cos_theta), dim=-1)   # Shape: T, 1, 2
    rz = torch.cat((rz1, rz2), dim=1)  # Shape: T, 2, 2

    return rz



def random_rot(data_numpy, theta=0.3):
    """
    input: data_numpy: C,T,V,M (array)
    output: CTVM, tensor
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    if C == 3:
        rot = torch.zeros(3).uniform_(-theta, theta)
        rot = torch.stack([rot, ] * T, dim=0)
        rot = _rot3d(rot)  # T,3,3
    elif C == 2:
        rot = torch.zeros(1).uniform_(-theta, theta)
        rot = torch.stack([rot, ] * T, dim=0)
        rot = _rot2d(rot)  # T,2,2
    # import pdb;pdb.set_trace()
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy

def CenterNorm_25j(kps):
    """
    Transform the skeleton to the center of the first frame
    :param kp: torch.Size nc,M,T,V,C=3
    :return: centernromed keypoint 5dim
    """
    nc,M,T,V,C = kps.shape
    num_frames=T
    for idx_nc, kp in enumerate(kps):
        kp = kp.permute(1,0,2,3).reshape(T,-1) #TMVC
        kp = kp.numpy()
        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(kp[i, :75] != 0):
                break
            i += 1
        origin = np.copy(kp[i, 3:6])  # new origin: joint-2
        for f in range(num_frames):
            kp[f] -= np.tile(origin, 50)
        kp = torch.tensor(kp)
        kp = kp.reshape(T, M, V, C).permute(1,0,2,3)  # M,T,V,C
        kps[idx_nc] = kp  # Update
    return kps
def CenterNorm_17j(kps):
    """
    Transform the skeleton to the center of the first frame
    :param kp: torch.Size nc,M,T,V,C=3
    :return: centernromed keypoint 5dim
    """
    nc,M,T,V,C = kps.shape
    num_frames=T
    for idx_nc, kp in enumerate(kps):
        kp = kp.permute(1,0,2,3).reshape(T,-1) #TMVC
        kp = kp.numpy()
        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(kp[i, :34] != 0):
                break
            i += 1
        origin = np.copy(kp[i, 0:2])  # new origin: joint-2
        for f in range(num_frames):
            kp[f] -= np.tile(origin, 34)
        kp = torch.tensor(kp)
        kp = kp.reshape(T, M, V, C).permute(1,0,2,3)  # M,T,V,C
        kps[idx_nc] = kp  # Update
    return kps

def Scale(kps,scale_range):
    """
    scale the keypints
    :param kp: torch.Size nc,M,T,V,C=3
    :return: scaled keypoint 5dim
    """
    nc, M, T, V, C = kps.shape
    kps = kps.numpy()
    scales = np.random.uniform(scale_range[0], scale_range[1], size=(C,))
    for c in range(C):
        kps[..., c] *= scales[c]
    kps = torch.tensor(kps)
    return kps

def Flip3d(kps):
    nc, M, T, V, C = kps.shape
    data = kps
    do_flip_x = torch.rand(1).item() < 0.5
    if do_flip_x:
        data[:, :, :, :, 0] = -data[:, :, :, :, 0]
    do_flip_y = torch.rand(1).item() < 0.5
    if do_flip_y:
        data[:, :, :, :, 1] = -data[:, :, :, :, 1]
    return data

class ComposeX:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')

        for a in self.augmentations:
            data = a(data)
            if data is None:
                return None
        return data

class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta information.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DC(meta, cpu_only=True)

        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data


class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class MergeSkeFeat:
    def __init__(self, feat_list=['keypoint'], target='keypoint', axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis. """

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results


class GenSkeFeat:
    def __init__(self, dataset='nturgb+d', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(Rename({'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)


class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 is_test=False,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.is_test = is_test
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        if len(deprecated_kwargs):
            warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)
        # pdb.set_trace()
        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        # pdb.set_trace()
        num_frames = results['total_frames']

        if self.is_test:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results


class UniformSampleGivenFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 given_len,
                 is_test=False,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.given_len = given_len
        self.num_clips = num_clips
        self.seed = seed
        self.is_test = is_test
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        if len(deprecated_kwargs):
            warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)
        # pdb.set_trace()
        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        # pdb.set_trace()
        num_frames = results['total_frames']

        if self.is_test:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)
            inds = np.concatenate([np.arange(x, x+self.given_len) for x in inds], axis=0)
            inds = np.clip(inds, 0, num_frames-1)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.clip_len * self.given_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results



class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "flip_direction".
    The Flip augmentation should be placed after any cropping / reshaping
    augmentations, to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    def _flip_imgs(self, imgs, modality):
        _ = [imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = self.iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def iminvert(img):
        """Invert (negate) an image.

        Args:
            img (ndarray): Image to be inverted.

        Returns:
            ndarray: The inverted image.
        """
        return np.full_like(img, 255) - img


    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'keypoint' in results:
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'], modality)
            if 'keypoint' in results:
                kp = results['keypoint']
                kpscore = results.get('keypoint_score', None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results['keypoint'] = kp
                if 'keypoint_score' in results:
                    results['keypoint_score'] = kpscore

        if 'gt_bboxes' in results and flip:
            assert self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map})')
        return repr_str


class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):
        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)
        return results


class PreNormalize2D:
    """Normalize the range of keypoint values. """

    def __init__(self, img_shape=(1080, 1920), threshold=0.01, mode='auto'):
        self.threshold = threshold
        # Will skip points with score less than threshold
        self.img_shape = img_shape
        self.mode = mode
        assert mode in ['fix', 'auto']

    def __call__(self, results):
        if 'keypoint' not in results: # for k400
            kp_path = results['raw_file']
            with open(kp_path, 'rb') as f:
                kp_dict = pickle.load(f)
            kp = kp_dict[results['frame_dir']]['keypoint']    #TMVC


            frame_inds = results['frame_inds']
            box_scores = results['box_score']

            selected_indices = []
            unique_frame_inds = np.unique(frame_inds)
            frame_num = len(unique_frame_inds)
            results['total_frames'] = frame_num
            M_ = 2
            TM,V,C = kp.shape
            for frame_ind in unique_frame_inds:

                indices_with_frame_ind = np.where(frame_inds == frame_ind)[0]

                sorted_indices = indices_with_frame_ind[np.argsort(box_scores[indices_with_frame_ind])[::-1]]
                if len(sorted_indices) >= M_:
                    selected_indices.extend(sorted_indices[:M_])
                elif len(sorted_indices) == 1:
                    sorted_indices = np.repeat(sorted_indices, M_)
                    selected_indices.extend(sorted_indices)
            kp = kp[selected_indices].reshape(frame_num,M_,V,C).transpose(1,0,2,3)
            results['keypoint'] = kp


        mask, maskout, keypoint = None, None, results['keypoint'].astype(np.float32)
        if 'keypoint_score' in results:
            keypoint_score = results.pop('keypoint_score').astype(np.float32)
            keypoint = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)

        if keypoint.shape[-1] == 3:
            mask = keypoint[..., 2] > self.threshold
            maskout = keypoint[..., 2] <= self.threshold

        if self.mode == 'auto':
            if mask is not None:
                if np.sum(mask):
                    x_max, x_min = np.max(keypoint[mask, 0]), np.min(keypoint[mask, 0])
                    y_max, y_min = np.max(keypoint[mask, 1]), np.min(keypoint[mask, 1])
                else:
                    x_max, x_min, y_max, y_min = 0, 0, 0, 0
            else:
                x_max, x_min = np.max(keypoint[..., 0]), np.min(keypoint[..., 0])
                y_max, y_min = np.max(keypoint[..., 1]), np.min(keypoint[..., 1])
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                keypoint[..., 0] = (keypoint[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                keypoint[..., 1] = (keypoint[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
        else: #fix
            h, w = results.get('img_shape', self.img_shape)
            keypoint[..., 0] = (keypoint[..., 0] - (w / 2)) / (w / 2)
            keypoint[..., 1] = (keypoint[..., 1] - (h / 2)) / (h / 2)

        if maskout is not None:
            keypoint[..., 0][maskout] = 0
            keypoint[..., 1][maskout] = 0
        results['keypoint'] = keypoint

        return results

class PreNormalize3D:
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z). Codes adapted from https://github.com/lshiwjx/2s-AGCN. """

    def __init__(self, zaxis=[0, 1], xaxis=[8, 4], align_spine=True, align_center=True):
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __call__(self, results):
        # pdb.set_trace()

        skeleton = results['keypoint']
        total_frames = results.get('total_frames', skeleton.shape[1])

        M, T, V, C = skeleton.shape
        assert T == total_frames
        if skeleton.sum() == 0:
            return results

        index0 = [i for i in range(T) if not np.all(np.isclose(skeleton[0, i], 0))]

        assert M in [1, 2]
        if M == 2:
            index1 = [i for i in range(T) if not np.all(np.isclose(skeleton[1, i], 0))]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        results['keypoint'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results

class GenSkeFeat:
    def __init__(self, dataset='nturgb+d', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(Rename({'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = ComposeX(ops)

    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)

class FormatGCNInput3D:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, window=None, rotate=True, mode='loop'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode
        self.window = window
        self.rotate = rotate

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        # if 'keypoint_score' in results:
        #     keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        # padding to fixed M
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]
        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        # random_rot
        if self.rotate:
            keypoint = keypoint.transpose(3,1,2,0) # MTVC->CTVM
            keypoint = random_rot(keypoint)
        else: # test, no rotate
            keypoint = keypoint.transpose(3,1,2,0) # MTVC->CTVM,array
            keypoint = torch.from_numpy(keypoint) # array->tensor

        nc = results.get('num_clips', 1)

        # intorpolate the T
        # resize
        if self.window:
            if nc == 1:
                data = keypoint
                C,T,V,M = data.shape
                data = torch.tensor(data, dtype=torch.float)
                data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, T)
                data = data[None, None, :, :]
                data = F.interpolate(data, size=(C * V * M, self.window), mode='bilinear',
                                     align_corners=False).squeeze()  # could perform both up sample and down sample
                data = data.contiguous().view(C, V, M, self.window).permute(0, 3, 1, 2).contiguous().numpy()
                keypoint = data
            else:

                data = keypoint
                C,T,V,M = data.shape
                data = data.reshape((C, nc, T // nc, V, M)).permute(0,1,3,4,2).reshape(-1,T//nc)
                data = torch.tensor(data, dtype=torch.float)
                data = data[None, None, :, :]
                data = F.interpolate(data, size=(C * V * M * nc, self.window), mode='bilinear',
                                     align_corners=False).squeeze()  # could perform both up sample and down sample
                data = data.contiguous().view(C, nc, V, M, self.window).permute(0, 1, 4, 2, 3).contiguous().view(C,nc*self.window,V,M).contiguous().numpy()
                keypoint = data


        if isinstance(keypoint, torch.Tensor):
            keypoint = keypoint.permute(3,1,2,0) # CTVM->MTVC
            keypoint = keypoint.numpy()
        elif isinstance(keypoint, np.ndarray):
            keypoint = keypoint.transpose(3, 1, 2, 0)  # CTVM->MTVC

        M, T, V, C = keypoint.shape
        # keypoint.transpose(3,1,2,0)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results # nc,M,T,V,C=3

class FormatGCNInput2D:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, window=None, rotate=True, mode='loop'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode
        self.window = window
        self.rotate = rotate

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)


        # M T V C
        if keypoint.shape[0] < self.num_person: # contain the case when M=0
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person: #M,T,V,C
            # # sample by confidence-sum
            M,T,V,C = keypoint.shape
            confidence = keypoint[..., -1] # M,T,V
            confidences = np.sum(confidence, axis=(1, 2)) # (M, )

            sorted_indices = np.argsort(confidences, axis=0)

            top_confi_indices = sorted_indices[-self.num_person:]
            keypoint = keypoint[top_confi_indices,:, :, :]


        # random_rot
        if self.rotate:
            keypoint = keypoint.transpose(3,1,2,0) # MTVC->CTVM
            keypoint = random_rot(keypoint)
        else: # test, no rotate
            keypoint = keypoint.transpose(3,1,2,0) # MTVC->CTVM,array
            keypoint = torch.from_numpy(keypoint) # array->tensor

        nc = results.get('num_clips', 1)

        # intorpolate the T
        # resize
        if self.window:
            if nc == 1:
                data = keypoint
                C, T, V, M = data.shape
                data = torch.tensor(data, dtype=torch.float)
                data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, T)
                data = data[None, None, :, :]
                data = F.interpolate(data, size=(C * V * M, self.window), mode='bilinear',
                                     align_corners=False).squeeze()  # could perform both up sample and down sample
                data = data.contiguous().view(C, V, M, self.window).permute(0, 3, 1, 2).contiguous().numpy()
                keypoint = data
            else:
                data = keypoint
                C, T, V, M = data.shape
                data = data.reshape((C, nc, T // nc, V, M)).permute(0, 1, 3, 4, 2).reshape(-1, T // nc)
                data = torch.tensor(data, dtype=torch.float)
                data = data[None, None, :, :]
                data = F.interpolate(data, size=(C * V * M * nc, self.window), mode='bilinear',
                                     align_corners=False).squeeze()  # could perform both up sample and down sample
                data = data.contiguous().view(C, nc, V, M, self.window).permute(0, 1, 4, 2, 3).contiguous().view(C,nc * self.window,V,M).contiguous().numpy()
                keypoint = data

        if isinstance(keypoint, torch.Tensor):
            keypoint = keypoint.permute(3,1,2,0) # CTVM->MTVC
            keypoint = keypoint.numpy()
        elif isinstance(keypoint, np.ndarray):
            keypoint = keypoint.transpose(3, 1, 2, 0)  # CTVM->MTVC
        M, T, V, C = keypoint.shape

        assert C == 3
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results # nc,M,T,V,C=2


class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str

class FormatGCNInputSkelter:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        if keypoint.ndim == 4:
            # M T V C
            if keypoint.shape[0] < self.num_person:
                pad_dim = self.num_person - keypoint.shape[0]
                pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
                keypoint = np.concatenate((keypoint, pad), axis=0)
                if self.mode == 'loop' and keypoint.shape[0] == 1:
                    for i in range(1, self.num_person):
                        keypoint[i] = keypoint[0]

            elif keypoint.shape[0] > self.num_person:
                keypoint = keypoint[:self.num_person]

            M, T, V, C = keypoint.shape
            nc = results.get('num_clips', 1)
            assert T % nc == 0
            keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        elif keypoint.ndim == 5:
            # num_clips, M, T, V, C
            num_clips = keypoint.shape[0]
            if keypoint.shape[1] < self.num_person:
                pad_dim = self.num_person - keypoint.shape[1]
                pad = np.zeros((num_clips, pad_dim, ) + keypoint.shape[2:], dtype=keypoint.dtype)
                keypoint = np.concatenate((keypoint, pad), axis=1)
                if self.mode == 'loop' and keypoint.shape[1] == 1:
                    for i in range(1, self.num_person):
                        keypoint[:, i] = keypoint[: ,0]

                elif keypoint.shape[1] > self.num_person:
                    keypoint = keypoint[:, : self.num_person]

        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str


class RandomScale:

    def __init__(self, scale=0.2):
        assert isinstance(scale, tuple) or isinstance(scale, float)
        self.scale = scale

    def __call__(self, results):
        skeleton = results['keypoint']
        scale = self.scale
        if isinstance(scale, float):
            scale = (scale, ) * skeleton.shape[-1]
        assert len(scale) == skeleton.shape[-1]
        scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
        results['keypoint'] = skeleton * scale
        return results

class RandomGaussianNoise:

    def __init__(self, sigma=0.01, base='frame', shared=False):
        assert isinstance(sigma, float)
        self.sigma = sigma
        self.base = base
        self.shared = shared
        assert self.base in ['frame', 'video']
        if self.base == 'frame':
            assert not self.shared

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape
        skeleton = skeleton.reshape(-1, V, C)
        ske_min, ske_max = skeleton.min(axis=1), skeleton.max(axis=1)
        # MT * C
        flag = ((ske_min ** 2).sum(axis=1) > EPS)
        # MT
        if self.base == 'frame':
            norm = np.linalg.norm(ske_max - ske_min, axis=1) * flag
            # MT
        elif self.base == 'video':
            assert np.sum(flag)
            ske_min, ske_max = ske_min[flag].min(axis=0), ske_max[flag].max(axis=0)
            # C
            norm = np.linalg.norm(ske_max - ske_min)
            norm = np.array([norm] * (M * T)) * flag
        # MT * V
        if self.shared:
            noise = np.random.randn(V) * self.sigma
            noise = np.stack([noise] * (M * T))
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            random_vec = np.concatenate([random_vec] * (M * T), axis=-1)
        else:
            noise = np.random.randn(M * T, V) * self.sigma
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, M * T * V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            # C * MTV
        random_vec = random_vec * noise.reshape(-1)
        # C * MTV
        random_vec = (random_vec.T).reshape(M, T, V, C)
        results['keypoint'] = skeleton.reshape(M, T, V, C) + random_vec
        return results

class RandomRot:

    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
        results['keypoint'] = np.einsum('ab,mtvb->mtva', rot_mat, skeleton)

        return results

class PoseCompact:
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required Keys:

        - keypoint
        - img_shape

    Modified Keys:

        - img_shape
        - keypoint

    Added Keys:

        - crop_quadruple

    Args:
        padding (float): The padding size. Defaults to 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Defaults to 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Defaults to None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Defaults to True.
    """

    def __init__(self,
                 padding=0.25,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def __call__(self, results):
        """Convert the coordinates of keypoints to make it more compact.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple) or isinstance(scale, list):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            self.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def imresize(
        img: np.ndarray,
        size: Tuple[int, int],
        return_scale: bool = False,
        interpolation: str = 'bilinear',
        out: Optional[np.ndarray] = None,
        backend: Optional[str] = None
    ) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
        """Resize image to a given size.

        Args:
            img (ndarray): The input image.
            size (tuple[int]): Target size (w, h).
            return_scale (bool): Whether to return `w_scale` and `h_scale`.
            interpolation (str): Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
                backend, "nearest", "bilinear" for 'pillow' backend.
            out (ndarray): The output destination.
            backend (str | None): The image resize backend type. Options are `cv2`,
                `pillow`, `None`. If backend is None, the global imread_backend
                specified by ``mmcv.use_backend()`` will be used. Default: None.

        Returns:
            tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
        """
        h, w = img.shape[:2]
        if backend is None:
            backend = imread_backend
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                            f"Supported backends are 'cv2', 'pillow'")

        if backend == 'pillow':
            assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
            resized_img = np.array(pil_image)
        else:
            resized_img = cv2.resize(
                img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale


    @staticmethod
    def _resize_kps(kps, scale_factor):
        """Static method for resizing keypoint."""
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            if 'imgs' in results:
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Static method for cropping keypoint."""
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Static method for cropping images."""
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str



class RandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str

class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required Keys:

        - keypoint
        - keypoint_score (optional)
        - img_shape

    Added Keys:

        - imgs (optional)
        - heatmap_imgs (optional)

    Args:
        sigma (float): The sigma of the generated gaussian map.
            Defaults to 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Defaults to True.
        with_kp (bool): Generate pseudo heatmaps for keypoints.
            Defaults to True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Defaults to False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Defaults to ``((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                         (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                         (13, 15), (6, 12), (12, 14), (14, 16), (11, 12))``,
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Defaults to False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Defaults to (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Defaults to (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Defaults to (0, 2, 4, 5, 6, 10, 11, 12),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Defaults to (1, 3, 7, 8, 9, 13, 14, 15),
            which is right limbs of skeletons we defined for COCO-17p.
        scaling (float): The ratio to scale the heatmaps. Defaults to 1.
    """

    def __init__(self,
                 sigma: float = 0.6,
                 use_score: bool = True,
                 with_kp: bool = True,
                 with_limb: bool = False,
                 skeletons: Tuple[Tuple[int]] = ((0, 1), (0, 2), (1, 3),
                                                 (2, 4), (0, 5), (5, 7),
                                                 (7, 9), (0, 6), (6, 8),
                                                 (8, 10), (5, 11), (11, 13),
                                                 (13, 15), (6, 12), (12, 14),
                                                 (14, 16), (11, 12)),
                 double: bool = False,
                 left_kp: Tuple[int] = (1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp: Tuple[int] = (2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb: Tuple[int] = (0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb: Tuple[int] = (1, 3, 7, 8, 9, 13, 14, 15),
                 scaling: float = 1.) -> None:

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling

    def generate_a_heatmap(self, arr: np.ndarray, centers: np.ndarray,
                           max_values: np.ndarray) -> None:
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < self.eps:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = \
                np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr: np.ndarray, starts: np.ndarray,
                                ends: np.ndarray, start_values: np.ndarray,
                                end_values: np.ndarray) -> None:
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint
                in the corresponding limbs. Shape: M.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None])
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = \
                np.maximum(arr[min_y:max_y, min_x:max_x], patch)

    def generate_heatmap(self, arr: np.ndarray, kps: np.ndarray,
                         max_values: np.ndarray) -> None:
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame.
                Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint.
                Shape: M * V.
        """

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends,
                                             start_values, end_values)

    def gen_an_aug(self, results: Dict) -> np.ndarray:
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            np.ndarray: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint'].astype(np.float32)
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)

        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else \
                np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)
        return ret

    def __call__(self, results: Dict) -> Dict:
        """Generate pseudo heatmaps based on joint coordinates and confidence.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        heatmap = self.gen_an_aug(results)
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (
                self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        results[key] = heatmap
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp}, '
                    f'left_limb={self.left_limb}, '
                    f'right_limb={self.right_limb}, '
                    f'scaling={self.scaling})')
        return repr_str


class FormatShape:
    """Format final imgs shape to the given input_format.

    Required keys:
        - imgs (optional)
        - heatmap_imgs (optional)
        - num_clips
        - clip_len

    Modified Keys:
        - imgs (optional)
        - input_shape (optional)

    Added Keys:
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
                'NCTHW', 'NCHW', 'NCHW_Flow', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW_Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            imgs = np.stack([x_flow, y_flow], axis=-1)

            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x T x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = T x C
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


class UniformSample(UniformSampleFrames):
    pass

class NTUSSampling(UniformSampleFrames):

    def __init__(self, num_clips=1, clip_olen=60, step=30, clip_len=30, seed=255, num_skeletons=20):
        super().__init__(num_clips=num_clips, clip_len=clip_len, seed=seed)
        self.num_skeletons = num_skeletons
        assert num_skeletons >= 1 or num_skeletons == -1
        self.clip_olen = clip_olen
        self.step = step

    def sample_seq_meta(self, num_person, total_frames, test_mode):
        if total_frames <= self.clip_olen:
            return [(i, 0, total_frames) for i in range(num_person)]
        if test_mode:
            offset = 0
            np.random.seed(self.seed)
        else:
            offset = np.random.randint(0, min(self.step, total_frames - self.clip_olen))
        windows = []
        for start in range(offset, total_frames - self.step, self.step):
            break_flag = False
            end = start + self.clip_olen
            if end >= total_frames:
                start, end = total_frames - self.clip_olen, total_frames
                break_flag = True
            windows.extend([(i, start, end) for i in range(num_person)])
            if break_flag:
                break
        return windows

    @staticmethod
    def get_stinfo(skeleton, img_shape, frame_idx, total_frames):
        h, w = img_shape
        t_embed = frame_idx / total_frames
        score = np.mean(skeleton[:, -1])
        min_x, max_x = min(skeleton[:, 0]), max(skeleton[:, 0])
        min_y, max_y = min(skeleton[:, 1]), max(skeleton[:, 1])
        c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        l_x, l_y = (max_x - min_x) * .625, (max_y - min_y) * .625
        stinfo = np.array([
            (c_x - l_x) / w, (c_y - l_y) / h, (c_x + l_x) / w, (c_y + l_y) / h, t_embed, score
        ], dtype=np.float32)
        stinfo[stinfo < 0] = 0
        stinfo[stinfo > 1] = 1
        return stinfo

    def __call__(self, results):
        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        assert T == results['total_frames']
        test_mode = results.get('test_mode', False)
        seq_meta = self.sample_seq_meta(M, T, test_mode)

        kpt_ret, stinfos = [], []

        for item in seq_meta:
            i, start, end = item
            kpt_sub = keypoint[i, start:end]
            if results['test_mode']:
                indices = self._get_test_clips(end - start, self.clip_len)
            else:
                indices = self._get_train_clips(end - start, self.clip_len)
            # Aug, T, V, C
            indices = np.mod(indices, end - start)
            kpt_sub = kpt_sub[indices].reshape((self.num_clips, self.clip_len, *kpt_sub.shape[1:]))
            kpt_ret.append(kpt_sub)
            ct_frame = (end + start) // 2
            if 'img_shape' in results:
                stinfos.append(self.get_stinfo(keypoint[i, ct_frame], results['img_shape'], ct_frame, T))
            else:
                # Not applicable, use a fake stinfo
                stinfos.append(np.array([0, 0, 1, 1, 0.5, 1], dtype=np.float32))

        # Aug, M, T, V, C
        kpt_ret = np.stack(kpt_ret, axis=1)
        # M, 6
        stinfo = np.stack(stinfos)

        num_skeletons, all_skeletons = self.num_skeletons, kpt_ret.shape[1]
        if num_skeletons == -1:
            num_skeletons = all_skeletons

        if all_skeletons > num_skeletons:
            stinfo = np.tile(stinfo[None], (self.num_clips, 1, 1))
            if results['test_mode']:
                indices = self._get_test_clips(all_skeletons, num_skeletons)
            else:
                indices = self._get_train_clips(all_skeletons, num_skeletons)
            indices = indices.reshape((self.num_clips, num_skeletons))
            for i in range(self.num_clips):
                kpt_ret[i, :num_skeletons] = kpt_ret[i, indices[i]]
                stinfo[i, :num_skeletons] = stinfo[i, indices[i]]
            kpt_ret = kpt_ret[:, :num_skeletons]
            stinfo = stinfo[:, :num_skeletons]
        else:
            kpt_ret_new = np.zeros((self.num_clips, num_skeletons, *kpt_ret.shape[2:]), dtype=np.float32)
            stinfo_new = np.zeros((self.num_clips, num_skeletons, 6), dtype=np.float32)
            kpt_ret_new[:, :all_skeletons] = kpt_ret
            stinfo_new[:, :all_skeletons] = stinfo
            kpt_ret = kpt_ret_new
            stinfo = stinfo_new

        results['keypoint'] = kpt_ret
        results['stinfo'] = stinfo
        results['name'] = results['frame_dir']
        return results

class KineticsSSampling(UniformSampleFrames):

    def __init__(self,
                 num_clips=1,
                 clip_olen=60,
                 clip_len=30,
                 step=30,
                 seed=255,
                 num_skeletons=20,
                 squeeze=True,
                 iou_thre=0,
                 track_method='bbox'):

        super().__init__(num_clips=num_clips, clip_len=clip_len, seed=seed)
        nske = num_skeletons
        if isinstance(nske, int):
            assert nske >= 1
            nske = (nske, nske)
        else:
            assert len(nske) == 2 and 0 < nske[0] <= nske[1]
        self.min_skeletons, self.max_skeletons = nske
        self.clip_olen = clip_olen
        self.step = step
        assert track_method in ['bbox', 'score', 'bbox&score', 'bbox_propagate', 'bbox_still']
        self.track_method = track_method

        from mmdet.core import BboxOverlaps2D
        self.iou_calc = BboxOverlaps2D()
        assert iou_thre in list(range(6))
        self.iou_thre = iou_thre
        assert iou_thre >= 0
        self.squeeze = squeeze

    @staticmethod
    def auto_box(kpts, thre=0.3, expansion=1.25, default_shape=(320, 426)):
        # It can return None if the box is too small
        assert len(kpts.shape) == 3 and kpts.shape[-1] == 3
        score = kpts[..., 2]
        flag = score >= thre
        boxes = []
        img_h, img_w = default_shape
        for i, kpt in enumerate(kpts):
            remain = kpt[flag[i]]
            if remain.shape[0] < 2:
                boxes.append([0, 0, img_w, img_h])
            else:
                min_x, max_x = np.min(remain[:, 0]), np.max(remain[:, 0])
                min_y, max_y = np.min(remain[:, 1]), np.max(remain[:, 1])
                if max_x - min_x < 10 or max_y - min_y < 10:
                    boxes.append([0, 0, img_w, img_h])
                else:
                    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
                    lx, ly = (max_x - min_x) / 2 * expansion, (max_y - min_y) / 2 * expansion
                    min_x, max_x = cx - lx, cx + lx
                    min_y, max_y = cy - ly, cy + ly
                    boxes.append([min_x, min_y, max_x, max_y])
        return np.array(boxes).astype(np.float32)

    def sample_seq_meta(self, ind2info, total_frames, test_mode):
        if total_frames <= self.clip_olen:
            max_idx, max_val = -1, 0
            for i in ind2info:
                if len(ind2info[i]) > max_val:
                    max_val = len(ind2info[i])
                    max_idx = i
            return [(0, total_frames, max_idx, i) for i in range(max_val)]
        if test_mode:
            offset = 0
            np.random.seed(self.seed)
        else:
            offset = np.random.randint(0, min(self.step, total_frames - self.clip_olen))
        windows = []

        clip_olen = self.clip_olen
        for start in range(offset, total_frames - self.step, self.step):
            break_flag = False
            if start + clip_olen >= total_frames:
                start = total_frames - clip_olen
                break_flag = True

            center_frame = start + clip_olen // 2 if test_mode else np.random.randint(start, start + clip_olen)
            num_boxes = len(ind2info[center_frame])
            windows.extend([(start, start + clip_olen, center_frame, i) for i in range(num_boxes)])
            if break_flag:
                break
        return windows

    def track_pre_computed(self, ind2info, start, end, center_idx, box_idx, keypoint, prevnext):
        inst = ind2info[center_idx][box_idx]
        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        prev, nxt = prevnext

        for i in range(center_idx - 1, start - 1, -1):
            p = prev[inst]
            if p == 65535:
                break
            kpt_ret[i - start] = keypoint[p]
            inst = p
        inst = original_inst
        for i in range(center_idx + 1, end):
            n = nxt[inst]
            if n == 65535:
                break
            kpt_ret[i - start] = keypoint[n]
            inst = n
        return kpt_ret

    def track_by_score(self, ind2info, start, end, center_idx, box_idx, keypoint, score_rank):
        inst = ind2info[center_idx][box_idx]
        rank = None
        for k in score_rank[center_idx]:
            if score_rank[center_idx][k] == inst:
                rank = k
        assert rank is not None

        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)

        for i in range(start, end):
            if rank in score_rank[i]:
                cinst = score_rank[i][rank]
                kpt_ret[i - start] = keypoint[cinst]
        return kpt_ret

    def track_by_ious(self, ind2info, start, end, center_idx, box_idx, keypoint, ious):
        # We track by bounding box
        inst = ind2info[center_idx][box_idx]
        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        cur_box_id = box_idx

        for i in range(center_idx - 1, start - 1, -1):
            iou = ious[i].T
            cur_box_id = np.argmax(iou[cur_box_id])
            kpt_ret[i - start] = keypoint[ind2info[i][cur_box_id]]

        inst, cur_box_id = original_inst, box_idx

        for i in range(center_idx + 1, end):
            iou = ious[i - 1]
            cur_box_id = np.argmax(iou[cur_box_id])
            kpt_ret[i - start] = keypoint[ind2info[i][cur_box_id]]
        return kpt_ret

    def track_by_bbox(self, ind2info, start, end, center_idx, box_idx, keypoint, bbox):
        # We track by bounding box
        inst = ind2info[center_idx][box_idx]
        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        cur_box = bbox[inst]

        if self.track_method == 'bbox_propagate':
            for i in range(center_idx - 1, start - 1, -1):
                bboxes = torch.tensor([bbox[x] for x in ind2info[i]])
                cur_box_t = torch.tensor(cur_box)[None]
                ious = self.iou_calc(cur_box_t, bboxes)[0]
                idx = torch.argmax(ious)
                if ious[idx] < self.iou_thre / 10:
                    break
                kpt_ret[i - start] = keypoint[ind2info[i][idx]]
                cur_box = bbox[ind2info[i][idx]]
            inst = original_inst
            cur_box = bbox[inst]
            for i in range(center_idx + 1, end):
                bboxes = torch.tensor([bbox[x] for x in ind2info[i]])
                cur_box_t = torch.tensor(cur_box)[None]
                ious = self.iou_calc(cur_box_t, bboxes)[0]
                idx = torch.argmax(ious)
                if ious[idx] < self.iou_thre / 10:
                    break
                kpt_ret[i - start] = keypoint[ind2info[i][idx]]
                cur_box = bbox[ind2info[i][idx]]
        elif self.track_method == 'bbox_still':
            st_inst, ed_inst = min(ind2info[start]), max(ind2info[end - 1])
            bboxes = torch.tensor(bbox[st_inst: ed_inst + 1])
            cur_box_t = torch.tensor(cur_box)[None]
            ious = self.iou_calc(cur_box_t, bboxes)[0]
            for t in range(start, end):
                if t == center_idx:
                    continue
                box_st, box_ed = min(ind2info[t]), max(ind2info[t])
                box_idx_t = torch.argmax(ious[box_st - st_inst: box_ed - st_inst + 1]) + box_st
                if torch.max(ious[box_st - st_inst: box_ed - st_inst + 1]) >= self.iou_thre / 10:
                    kpt_ret[t - start] = keypoint[box_idx_t]
        return kpt_ret

    def track_by_bns(self, ind2info, start, end, center_idx, box_idx, keypoint, prevnext, score_rank):
        inst = ind2info[center_idx][box_idx]
        rank = None
        for k in score_rank[center_idx]:
            if score_rank[center_idx][k] == inst:
                rank = k
        assert rank is not None

        original_inst = inst
        kpt_shape = keypoint[0].shape
        kpt_ret = np.zeros((end - start, ) + kpt_shape, dtype=np.float32)
        kpt_ret[center_idx - start] = keypoint[inst]
        prev, nxt = prevnext

        for i in range(center_idx - 1, start - 1, -1):
            p = prev[inst]
            if p == 65535:
                if rank in score_rank[i]:
                    p = score_rank[i][rank]
                else:
                    break
            kpt_ret[i - start] = keypoint[p]
            inst = p

        inst = original_inst
        for i in range(center_idx + 1, end):
            n = nxt[inst]
            if n == 65535:
                if rank in score_rank[i]:
                    n = score_rank[i][rank]
                else:
                    break
            kpt_ret[i - start] = keypoint[n]
            inst = n
        return kpt_ret

    def _get_score_rank(self, keypoint, ind2info):
        scores = keypoint[..., 2].sum(axis=-1)
        score_rank = dict()
        for find in ind2info:
            score_rank[find] = dict()
            seg = ind2info[find]
            s, e = min(seg), max(seg) + 1
            score_sub = scores[s: e]
            order_sub = (-score_sub).argsort()
            rank_sub = order_sub.argsort()
            for i in range(e - s):
                idx = s + i
                score_rank[find][rank_sub[i]] = idx
        return score_rank

    def __call__(self, results):

        if results['keypoint'].dtype == np.float16:
            results['keypoint'] = results['keypoint'].astype(np.float32)

        if 'bbox' in results and results['bbox'].dtype == np.float16:
            results['bbox'] = results['bbox'].astype(np.float32)

        keypoint = results['keypoint']
        ske_frame_inds = results['ske_frame_inds']
        total_frames = results['total_frames']
        test_mode = results['test_mode']

        def mapinds(ske_frame_inds):
            uni = np.unique(ske_frame_inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in ske_frame_inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            ske_frame_inds = mapinds(ske_frame_inds)
            total_frames = np.max(ske_frame_inds) + 1
            results['ske_frame_inds'], results['total_frames'] = ske_frame_inds, total_frames

        kpt_shape = keypoint[0].shape

        assert keypoint.shape[0] == ske_frame_inds.shape[0]

        h, w = results['img_shape']
        bbox = results.get('bbox', self.auto_box(keypoint, default_shape=results['img_shape']))
        bbox = bbox.astype(np.float32)
        bbox[:, 0::2] /= w
        bbox[:, 1::2] /= h
        bbox = np.clip(bbox, 0, 1)

        ind2info = defaultdict(list)
        for i, find in enumerate(ske_frame_inds):
            ind2info[find].append(i)

        seq_meta = self.sample_seq_meta(ind2info, total_frames, test_mode)

        kpt_ret, stinfos = [], []
        ious = results.pop('ious', None)

        if ious is not None:
            assert len(ious) == len(set(ske_frame_inds)) - 1

        if 'score' in self.track_method:
            score_rank = results.pop('score_rank', self._get_score_rank(keypoint, ind2info))

        prevnext = results.pop('iouthr_pn', None)
        if prevnext is not None:
            assert prevnext.shape == (6, 2, keypoint.shape[0])
            prevnext = prevnext[self.iou_thre]

        # In train mode, we use max_skeletons
        if not test_mode and len(seq_meta) > self.max_skeletons:
            assert self.num_clips == 1
            indices = self._get_train_clips(len(seq_meta), self.max_skeletons)
            seq_meta = [seq_meta[i] for i in indices]

        if self.track_method == 'bbox' and ious is None and prevnext is None:
            self.track_method = 'bbox_propagate'
        if self.track_method in ['bbox_propagate', 'bbox_still']:
            prevnext, ious = None, None

        for item in seq_meta:
            start, end, center_idx, box_idx = item
            cur_ske_idx = ind2info[center_idx][box_idx]
            cur_kpt, cur_box = keypoint[cur_ske_idx], bbox[cur_ske_idx]
            stinfo = np.array(list(cur_box) + [center_idx / total_frames, np.mean(cur_kpt[:, 2])], dtype=np.float32)
            stinfos.append(stinfo)
            if self.track_method in ['bbox_propagate', 'bbox_still']:
                kpt = self.track_by_bbox(ind2info, start, end, center_idx, box_idx, keypoint, bbox)
            elif self.track_method == 'score':
                kpt = self.track_by_score(ind2info, start, end, center_idx, box_idx, keypoint, score_rank)
            elif self.track_method == 'bbox':
                if prevnext is not None:
                    kpt = self.track_pre_computed(ind2info, start, end, center_idx, box_idx, keypoint, prevnext)
                elif ious is not None:
                    kpt = self.track_by_ious(ind2info, start, end, center_idx, box_idx, keypoint, ious)
            elif self.track_method == 'bns':
                kpt = self.track_by_bns(ind2info, start, end, center_idx, box_idx, keypoint, prevnext, score_rank)

            if test_mode:
                indices = self._get_test_clips(end - start, self.clip_len)
            else:
                indices = self._get_train_clips(end - start, self.clip_len)

            indices = np.mod(indices, end - start)
            kpt = kpt[indices].reshape((self.num_clips, self.clip_len, *kpt_shape))
            kpt_ret.append(kpt)

        # Aug, Skeletons, Clip_len, V, C
        kpt_ret = np.stack(kpt_ret, axis=1)
        # Skeletons, 6
        stinfo_old = np.stack(stinfos)

        min_ske, max_ske, all_skeletons = self.min_skeletons, self.max_skeletons, kpt_ret.shape[1]
        num_ske = np.clip(all_skeletons, min_ske, max_ske) if test_mode else max_ske
        keypoint = np.zeros((self.num_clips, num_ske) + kpt_ret.shape[2:], dtype=np.float32)
        stinfo = np.zeros((self.num_clips, num_ske, 6), dtype=np.float32)

        if test_mode:
            if all_skeletons < num_ske:
                keypoint[:, :all_skeletons] = kpt_ret
                stinfo[:, :all_skeletons] = stinfo_old
            elif all_skeletons > num_ske:
                stinfo_old = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                indices = self._get_test_clips(all_skeletons, num_ske)
                indices = indices.reshape((self.num_clips, num_ske))
                for i in range(self.num_clips):
                    keypoint[i] = kpt_ret[i, indices[i]]
                    stinfo[i] = stinfo_old[i, indices[i]]
            else:
                stinfo = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                keypoint = kpt_ret
        else:
            # only use max_ske
            if all_skeletons > num_ske:
                stinfo_old = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                indices = self._get_train_clips(all_skeletons, num_ske)
                indices = indices.reshape((self.num_clips, num_ske))
                for i in range(self.num_clips):
                    keypoint[i] = kpt_ret[i, indices[i]]
                    stinfo[i] = stinfo_old[i, indices[i]]
            else:
                keypoint[:, :all_skeletons] = kpt_ret
                stinfo[:, :all_skeletons] = stinfo_old

        results['keypoint'] = keypoint
        results['stinfo'] = stinfo
        results['name'] = results['frame_dir']
        return results

class AVASSampling(UniformSampleFrames):

    def __init__(self,
                 num_clips=1,
                 clip_olen=60,
                 clip_len=30,
                 seed=255,
                 num_skeletons=20,
                 rel_offset=0.75):
        super().__init__(num_clips=num_clips, clip_len=clip_len, seed=seed)
        self.clip_olen = clip_olen
        nske = num_skeletons
        if isinstance(nske, int):
            assert nske >= 1
            nske = (nske, nske)
        else:
            assert 0 < nske[0] <= nske[1]
        self.min_skeletons, self.max_skeletons = nske
        self.rel_offset = rel_offset

    def __call__(self, results):
        assert 'data' in results
        data = results.pop('data')
        test_mode = results['test_mode']

        kpts, stinfos, labels, names = [], [], [], []

        for i, frame in enumerate(data):
            for ske in frame:
                keypoint, name, label = ske['keypoint'], ske['name'], ske['label']
                box = ske['box']
                box_score = ske.get('box_score', 1.0)

                if len(keypoint.shape) == 4:
                    assert keypoint.shape[0] == 1
                    keypoint = keypoint[0]
                # keypoint.shape == (T, V, C)
                stinfo = np.array(list(box) + [i / len(data), box_score], dtype=np.float32)
                stinfo = np.clip(stinfo, 0, 1)
                stinfos.append(stinfo)
                names.append(name)
                labels.append(label)

                total_frames = keypoint.shape[0]
                if total_frames > self.clip_olen:
                    offset = (total_frames - self.clip_olen) // 2
                    if not test_mode:
                        offset *= (1 - self.rel_offset) + np.random.rand() * self.rel_offset * 2
                        offset = int(offset)
                    keypoint = keypoint[offset:offset + self.clip_olen]
                    total_frames = self.clip_olen

                assert total_frames >= self.clip_len
                if test_mode:
                    frame_inds = self._get_test_clips(total_frames, self.clip_len)
                else:
                    frame_inds = self._get_train_clips(total_frames, self.clip_len)
                assert len(frame_inds) == self.clip_len * self.num_clips
                keypoint = keypoint[frame_inds].reshape((self.num_clips, self.clip_len) + (keypoint.shape[1:]))
                # keypoint.shape == (num_clips, T, V, C)
                kpts.append(keypoint)

        kpts = np.stack(kpts, axis=1).astype(np.float32)  # kpts.shape == (num_clips, num_skeletons, T, V, C)
        labels = np.stack(labels)  # labels.shape == (num_skeletons, num_classes)
        stinfo_old = np.stack(stinfos).astype(np.float32)  # stinfo_old.shape = (num_skeletons, 6)
        names = np.stack(names)

        min_ske, max_ske, all_skeletons = self.min_skeletons, self.max_skeletons, kpts.shape[1]
        num_ske = np.clip(all_skeletons, min_ske, max_ske) if test_mode else max_ske
        keypoint = np.zeros((self.num_clips, num_ske) + kpts.shape[2:], dtype=np.float32)
        label = np.zeros((self.num_clips, num_ske, labels.shape[-1]), dtype=np.float32)
        stinfo = np.zeros((self.num_clips, num_ske, 6), dtype=np.float32)

        if test_mode:
            if all_skeletons > num_ske:
                indices = self._get_test_clips(all_skeletons, num_ske).reshape((self.num_clips, num_ske))
                new_names = []
                for i in range(self.num_clips):
                    keypoint[i] = kpts[i, indices[i]]
                    label[i] = labels[indices[i]]
                    stinfo[i] = stinfo_old[indices[i]]
                    new_names.append(names[indices[i]])
                names = np.stack(new_names)
            elif all_skeletons < num_ske:
                keypoint[:, :all_skeletons] = kpts
                label[:, :all_skeletons] = labels
                stinfo[:, :all_skeletons] = stinfo_old
                names = np.concatenate([names, ['NA'] * (num_ske - all_skeletons)])
                names = np.stack([names] * self.num_clips)
            else:
                keypoint = kpts
                label = np.tile(labels[None], (self.num_clips, 1, 1))
                stinfo = np.tile(stinfo_old[None], (self.num_clips, 1, 1))
                names = np.stack([names] * self.num_clips)
        else:
            if all_skeletons > num_ske:
                indices = self._get_train_clips(all_skeletons, num_ske)
                indices = indices.reshape((self.num_clips, num_ske))
                new_names = []
                for i in range(self.num_clips):
                    keypoint[i] = kpts[i, indices[i]]
                    label[i] = labels[indices[i]]
                    stinfo[i] = stinfo_old[indices[i]]
                    new_names.append(names[indices[i]])
                names = np.stack(new_names)
            else:
                keypoint[:, :all_skeletons] = kpts
                label[:, :all_skeletons] = labels
                stinfo[:, :all_skeletons] = stinfo_old
                names = np.concatenate([names, ['NA'] * (num_ske - all_skeletons)])
                names = np.stack([names] * self.num_clips)

        results['keypoint'] = keypoint
        results['label'] = label
        results['stinfo'] = stinfo
        results['name'] = names
        return results


def _combine_quadruple(a, b):
    return a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3]

def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'

def rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

def imflip_(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)

def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)
