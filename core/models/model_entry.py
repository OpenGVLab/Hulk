import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, cv2
import numpy as np
from core.utils import (add_task_specific, add_neck_specific, add_decoder_specific, add_backbone_specific,
                        add_aiov2_decoder_specific, add_aiov2_backbone_specific, add_aiov2_modality_specific, add_aiov2_task_specific)
from core.data.transforms.post_transforms import pose_pck_accuracy, flip_back, transform_preds

class model_entry(nn.Module):
    def __init__(self, backbone_module, neck_module, decoder_module):
        super(model_entry, self).__init__()
        self.backbone_module = backbone_module
        self.neck_module = neck_module
        self.decoder_module = decoder_module
        add_task_specific(self, False)
        add_backbone_specific(self.backbone_module, True)
        add_neck_specific(self.neck_module, True)
        add_decoder_specific(self.decoder_module, True)
        if hasattr(self.decoder_module, 'loss'):
            if hasattr(self.decoder_module.loss, 'classifier'):
                add_task_specific(self.decoder_module.loss, True)

    def forward(self, input_var, current_step):
        x = self.backbone_module(input_var) # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        x = self.neck_module(x)
        decoder_feature = self.decoder_module(x)
        return decoder_feature


class aio_entry_v2mae_shareneck(nn.Module):
    def __init__(self, backbone_module, patch_neck_module, label_neck_module, decoder_module,
                 patch_adapter_module=None, label_adapter_module=None,
                 patch_proj_module=None, label_proj_module=None,
                 modalities={}, kwargs={}):
        super().__init__()
        self.backbone_module = backbone_module
        self.decoder_module = decoder_module
        self.modalities = modalities
        self.kwargs = kwargs
        self.test_flag = self.kwargs.get('test_flag', None)
        self.flip_channels = self.kwargs.get('flip_channels', False)

        self.add_module('_'.join(['adapter', self.modalities['patch']]), patch_adapter_module)
        self.add_module('_'.join(['adapter', self.modalities['label']]), label_adapter_module)

        patch_adatper_name = 'self.adapter_{}'.format(self.modalities['patch'])
        label_adatper_name = 'self.adapter_{}'.format(self.modalities['label'])

        self.add_module('_'.join(['neck', 'patch']), patch_neck_module)
        self.add_module('_'.join(['neck', 'label']), label_neck_module)

        patch_neck_name = 'self.neck_patch'
        label_neck_name = 'self.neck_label'


        self.add_module('_'.join(['proj', self.modalities['patch']]), patch_proj_module)
        self.add_module('_'.join(['proj', self.modalities['label']]), label_proj_module)

        patch_proj_name = 'self.proj_{}'.format(self.modalities['patch'])
        label_proj_name = 'self.proj_{}'.format(self.modalities['label'])

        self.patch_adatper_name = patch_adatper_name
        self.label_adapter_name = label_adatper_name
        self.patch_neck_name = patch_neck_name
        self.label_neck_name = label_neck_name
        self.patch_proj_name = patch_proj_name
        self.label_proj_name = label_proj_name

        add_task_specific(self, False)

        # as using the add_module in nn.Module(), the module names are feasible,
        # here we use the eval() with the module name to represent the
        # "self.neck_rgb" module with eval("self.neck_rgb")

        # modality share is truly the task share, e.g., all pose datasets share a same task,
        # therefore, the modality shared parameters are used as the task tokens.
        add_aiov2_modality_specific(eval(patch_adatper_name), self.modalities['patch'], True,
                                    eval(patch_adatper_name).task_sp_list, eval(patch_adatper_name).modality_share_list)

        add_aiov2_modality_specific(eval(label_adatper_name), self.modalities['label'], True,
                                    eval(label_adatper_name).task_sp_list, eval(patch_adatper_name).modality_share_list)

        add_aiov2_modality_specific(eval(patch_proj_name), self.modalities['patch'], True,
                                    eval(patch_proj_name).task_sp_list, eval(patch_proj_name).modality_share_list)

        add_aiov2_modality_specific(eval(label_proj_name), self.modalities['label'], True,
                                    eval(label_proj_name).task_sp_list, eval(label_proj_name).modality_share_list)

        add_aiov2_backbone_specific(self.backbone_module, True, self.backbone_module.task_sp_list,
                                    self.backbone_module.neck_sp_list)
        add_aiov2_decoder_specific(self.decoder_module, True, self.decoder_module.task_sp_list,
                                   self.decoder_module.neck_sp_list, self.decoder_module.modality_share_list)

        #  setting the neck as the same as the backbone (all shared parameters)
        add_aiov2_decoder_specific(eval(patch_neck_name), True, self.backbone_module.task_sp_list,
                                    self.backbone_module.neck_sp_list)
        add_aiov2_decoder_specific(eval(label_neck_name), True, self.backbone_module.task_sp_list,
                                    self.backbone_module.neck_sp_list)

    def forward(self, input_var, current_step):
        if self.training:
            input_var = eval(self.patch_adatper_name)(input_var) # add key "patch tokens" to the dict
            input_var = eval(self.label_adapter_name)(input_var) # add key "label tokens" to the dict
            x = self.backbone_module(input_var) # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
            x = eval(self.patch_neck_name)(x)
            x = eval(self.label_neck_name)(x)

            decoder_feature = self.decoder_module(x)
            patch_outputs = eval(self.patch_proj_name)(decoder_feature)
            # import pdb;pdb.set_trace()
            label_outputs = eval(self.label_proj_name)(decoder_feature)
            output={}
            output['outputs'] = patch_outputs
            output['outputs'].update(label_outputs)
        else:
             # task_flag
            if self.test_flag is None:
                output = self.forward_default_test(input_var, current_step)
            elif self.test_flag == 'image_caption':
                output = self.forward_test_caption(input_var, current_step)
            elif self.test_flag == 'pose':
                output = self.forward_test_pose_bce(input_var, current_step)
            elif self.test_flag == 'par_flip':
                output = self.forward_test_par_flip(input_var, current_step)
            else:
                raise ValueError("test_flag ({}) is NOT supported!".format(self.test_flag))

        return output

    def forward_default_test(self, input_var, current_step):
        input_var = eval(self.patch_adatper_name)(input_var)  # add key "patch tokens" to the dict
        input_var = eval(self.label_adapter_name)(input_var)  # add key "label tokens" to the dict
        x = self.backbone_module(
            input_var)  # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        x = eval(self.patch_neck_name)(x)
        x = eval(self.label_neck_name)(x)

        decoder_feature = self.decoder_module(x)
        patch_outputs = eval(self.patch_proj_name)(decoder_feature)
        # import pdb;pdb.set_trace()
        label_outputs = eval(self.label_proj_name)(decoder_feature)
        output = {}
        output['pred'] = label_outputs
        output['pred_patch'] = patch_outputs

        return output

        # image caption test forward
    def forward_test_caption(self, input_var, current_step):
        # import pdb; pdb.set_trace()
        assert self.training is False, "forward_test_caption only supports for testing"
        input_var = eval(self.patch_adatper_name)(input_var) # add key "patch tokens" to the dict
        input_var = eval(self.label_adapter_name)(input_var) # add key "label tokens" to the dict
        x = self.backbone_module(input_var) # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        x = eval(self.patch_neck_name)(x)
        x = eval(self.label_neck_name)(x)
        # prepare for caption input
        bos_token_id, eos_token_ids, pad_token_id = 101, [102], 0
        batch_size, max_generate_len = x['input_id'].shape[0], x['input_id'].shape[1]
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device='cuda')
        cur_len = input_ids.shape[1]
        unfinished_sents, logprobs = [], []
        cur_unfinished = input_ids.new(batch_size).fill_(1)
        while cur_len < max_generate_len:
            pad_ids = torch.full((batch_size, max_generate_len - input_ids.shape[1]), pad_token_id, dtype=torch.long, device='cuda')
            x['cur_len'] = cur_len
            x['input_id'] = torch.cat([input_ids, pad_ids], dim=1)
            decoder_feature = self.decoder_module(x)
            patch_outputs = eval(self.patch_proj_name)(decoder_feature)
            label_outputs = eval(self.label_proj_name)(decoder_feature)
            outputs = label_outputs['logit']
            next_token_idx = cur_len - 1
            next_token_logits = outputs[:, next_token_idx, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)
            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            # 将刚预测出来的新token concat到input_ids
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long()) # tensor([1, 1, 1, 1, 1, 1], device='cuda:0')
            cur_len = cur_len + 1
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break
        # add eos_token_ids to unfinished sentences
        if cur_len == max_generate_len:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])
        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)
        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_generate_len - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)
        output = {}
        output['pred'] = input_ids
        output['pred_patch']= patch_outputs

        return output

        # pose estimation test forward
    def forward_test_pose_bce(self, input_var, current_step):
        input_var_flipped = copy.deepcopy(input_var)

        input_var = eval(self.patch_adatper_name)(input_var)
        input_var = eval(self.label_adapter_name)(input_var)
        x = self.backbone_module(
            input_var
        )
        x = eval(self.patch_neck_name)(x)
        x = eval(self.label_neck_name)(x)

        decoder_feature = self.decoder_module(x)

        patch_outputs = eval(self.patch_proj_name)(decoder_feature)
        label_outputs = eval(self.label_proj_name)(decoder_feature)

        # output_heatmap = label_outputs
        # import pdb;pdb.set_trace()
        input_var_flipped["image"] = input_var_flipped["image"].flip(3)
        input_var_flipped = eval(self.patch_adatper_name)(input_var_flipped)
        input_var_flipped = eval(self.label_adapter_name)(input_var_flipped)
        x_flipped = self.backbone_module(
            input_var_flipped
        )
        x_flipped = eval(self.patch_neck_name)(x_flipped)
        x_flipped = eval(self.label_neck_name)(x_flipped)

        decoder_feature_flipped = self.decoder_module(x_flipped)

        patch_outputs_flipped = eval(self.patch_proj_name)(decoder_feature_flipped)
        label_outputs_flipped = eval(self.label_proj_name)(decoder_feature_flipped)

        label_outputs_flipped['output_heatmap'] = flip_back(label_outputs_flipped['output_heatmap'], \
        flip_pairs=x.img_metas[0].data["flip_pairs"], target_type="GaussianHeatMap")

        output_heatmap = (label_outputs['output_heatmap'] + label_outputs_flipped['output_heatmap']) * 0.5

        keypoint_result = self.pose_decode(input_var["img_metas"], output_heatmap)  # default output_heatmap

        if 'pred_logits' in label_outputs:
            keypoint_result['pred_logits'] = label_outputs['pred_logits'].sigmoid().cpu().numpy()

        return keypoint_result

    def forward_test_par_flip(self, input_var, current_step):
        input_var_flipped = copy.deepcopy(input_var)

        input_var = eval(self.patch_adatper_name)(input_var)
        input_var = eval(self.label_adapter_name)(input_var)
        x = self.backbone_module(
            input_var)
        x = eval(self.patch_neck_name)(x)
        x = eval(self.label_neck_name)(x)
        decoder_feature = self.decoder_module(x)
        patch_outputs = eval(self.patch_proj_name)(decoder_feature)
        label_outputs = eval(self.label_proj_name)(decoder_feature)

        input_var_flipped["image"] = input_var_flipped["image"].flip(3) #torch.Size([16, 3, 480, 480])
        input_var_flipped = eval(self.patch_adatper_name)(input_var_flipped)
        input_var_flipped = eval(self.label_adapter_name)(input_var_flipped)
        x_flipped = self.backbone_module(
            input_var_flipped)
        x_flipped = eval(self.patch_neck_name)(x_flipped)
        x_flipped = eval(self.label_neck_name)(x_flipped)
        decoder_feature_flipped = self.decoder_module(x_flipped)
        patch_outputs_flipped = eval(self.patch_proj_name)(decoder_feature_flipped)
        label_outputs_flipped = eval(self.label_proj_name)(decoder_feature_flipped)
        flip_channels = np.array(self.flip_channels)
        left_channels = flip_channels[:, 0]
        right_channels = flip_channels[:, 1]

        # label_outputs_flipped = label_outputs_flipped.flip(3).changechannel
        for i in range(len(label_outputs)):
            ori = label_outputs[i]['sem_seg']
            flip = label_outputs_flipped[i]['sem_seg'].flip(2)  #torch.Size([20, 500, 334])
            flip_channeled = copy.deepcopy(flip)
            for idx, channel in enumerate(left_channels):
                flip_channeled[channel,:,:] = flip[right_channels[idx],:,:]
            for idx, channel in enumerate(right_channels):
                flip_channeled[channel,:,:] = flip[left_channels[idx],:,:]
            label_outputs[i]['sem_seg'] = (ori + flip_channeled) * 0.5


        output = {}
        output['pred'] = label_outputs
        output['pred_patch'] = patch_outputs

        return output

    def pose_decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0].data:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i].data['center']
            s[i, :] = img_metas[i].data['scale']
            image_paths.append(img_metas[i].data['image_file'])

            if 'bbox_score' in img_metas[i].data:
                score[i] = np.array(img_metas[i].data['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i].data['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=False,
            post_process='default',
            kernel=11,
            valid_radius_factor=0.0546875,
            use_udp=True,
            target_type="GaussianHeatMap")

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatMap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatMap' or 'CombinedTarget'.
            GaussianHeatMap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        assert target_type in ['GaussianHeatMap', 'CombinedTarget']
        if target_type == 'GaussianHeatMap':
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type == 'CombinedTarget':
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(np.int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatMap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps
