import logging
import os
import torch
from scipy.io import loadmat, savemat
import json_tricks as json
import numpy as np
import time

from collections import OrderedDict, defaultdict
from xtcocotools.cocoeval import COCOeval

from .seg_tester_dev import DatasetEvaluator
from .nms import oks_nms, soft_oks_nms
import core.data.datasets.images.resources as resources


class PoseEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.use_nms = config.evaluation.cfg.get('use_nms', True)
        self.soft_nms = config.evaluation.cfg.soft_nms
        self.nms_thr = config.evaluation.cfg.nms_thr
        self.oks_thr = config.evaluation.cfg.oks_thr
        self.vis_thr = config.evaluation.cfg.vis_thr
        self.cls_logits_vis_thr = config.evaluation.cfg.get('cls_logits_vis_thr', -1)
        self.no_rescoring = config.evaluation.cfg.get('no_rescoring', False)
        self.dataset = config.evaluation.cfg.dataset

        self.sigmas = np.array(config.evaluation.cfg.get('sigmas',
                                                         [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072,
                                                          0.072, 0.062,0.062, 0.107, 0.107, 0.087, 0.087, 0.089,
                                                          0.089]))  # default is coco-kp's sigmas
        self.use_area = config.evaluation.cfg.get('use_area', True)

        self.interval = config.evaluation.cfg.interval
        self.metric = config.evaluation.cfg.metric
        self.key_indicator = config.evaluation.cfg.key_indicator

        self.ann_info = {}
        self.ann_info['num_joints'] = config.dataset.kwargs.data_cfg['num_joints']
        self.config=config

    def reset(self):
        self.results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        # for input, output in zip(inputs, outputs):
        self.results.append(outputs)

        #  note: sync if multi-gpu

    def evaluate(self):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(dict))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_paths (list[str]): For example, ['data/coco/val2017
                    /000000393226.jpg']
                :heatmap (np.ndarray[N, K, H, W]): model output heatmap
                :bbox_id (list(int)).
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = self.metric if isinstance(self.metric, list) else [self.metric]
        allowed_metrics = ['mAP', "PCK", 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        assert self._output_dir

        os.makedirs(self._output_dir, exist_ok=True)
        res_file = os.path.join(self._output_dir, f'result_keypoints-{time.time()}.json')

        kpts = defaultdict(list)

        for output in self.results:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.dataset.name2id[image_paths[i][len(self.dataset.img_prefix):]]
                img_dict = {
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i], }
                if 'pred_logits' in output:
                    img_dict['pred_logits']=output['pred_logits'][i]
                kpts[image_id].append(img_dict)
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        if not self.no_rescoring:
            num_joints = self.ann_info['num_joints']
            vis_thr = self.vis_thr
            oks_thr = self.oks_thr
            valid_kpts = []
            for image_id in kpts.keys():
                img_kpts = kpts[image_id]
                for n_p in img_kpts:
                    box_score = n_p['score']
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

                if self.use_nms:
                    nms = soft_oks_nms if self.soft_nms else oks_nms
                    keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                    valid_kpts.append([img_kpts[_keep] for _keep in keep])
                else:
                    valid_kpts.append(img_kpts)

            self._write_coco_keypoint_results(valid_kpts, res_file)
        else:
            # import pdb;pdb.set_trace()
            self._write_keypoint_results(kpts, res_file)
        if 'mAP' in metrics:
            info_str = self._do_python_keypoint_eval(res_file)
            results = OrderedDict({"key_point": info_str})
            self._logger.info(results)
        if 'PCK' in metrics or 'EPE' in metrics:
            results = self._report_metric(res_file, metrics)

        # import pdb;pdb.set_trace()

        if self.cls_logits_vis_thr>=0 and 'pred_logits' in self.results[0] and not self.no_rescoring:
            print(f"Test with CLS logits and new vis_threshold {self.cls_logits_vis_thr}")

            valid_kpts = []
            vis_thr = self.cls_logits_vis_thr
            # import pdb;pdb.set_trace()
            for image_id in kpts.keys():
                img_kpts = kpts[image_id]
                for n_p in img_kpts:
                    box_score = n_p['score']
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2] * n_p['pred_logits'][n_jt]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

                if self.use_nms:
                    nms = soft_oks_nms if self.soft_nms else oks_nms
                    keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                    valid_kpts.append([img_kpts[_keep] for _keep in keep])
                else:
                    valid_kpts.append(img_kpts)

            self._write_coco_keypoint_results(valid_kpts, res_file)

            info_str = self._do_python_keypoint_eval(res_file)
            results = OrderedDict({"key_point": info_str})
            self._logger.info(results)

        os.remove(res_file)
        print(f"{res_file} deleted")
        return results

    def eval_cls_logits(self, kpts, vis_thr=0.05, ):
        valid_kpts = []
        num_joints = self.ann_info['num_joints']
        oks_thr = self.oks_thr
        # import pdb;pdb.set_trace()
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2] * n_p['pred_logits'][n_jt]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)
        res_file = os.path.join(self._output_dir, f'result_keypoints-{time.time()}.json')
        self._write_coco_keypoint_results(valid_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        results = OrderedDict({"key_point": info_str})
        os.remove(res_file)
        print(f"{res_file} deleted")
        return results

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.
        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.
        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = {}

        with open(res_file, 'r') as fin:
            preds = json.load(fin)

        assert len(preds) == len(self.dataset.db)

        outputs = []
        gts = []
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []
        # import pdb;
        # pdb.set_trace()
        for _, item in zip(preds, self.dataset.db):
            # p_ = preds[pred]
            pred = preds[str(self.dataset.name2id[item['image_file'].replace(self.dataset.img_prefix, '')])][0]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            # if 'PCKh' in metrics:
            #     head_box_thr = item['head_size']
            #     threshold_head_box.append(
            #         np.array([head_box_thr, head_box_thr]))
            # box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])
        results = OrderedDict()
        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            results["PCK_key_point"]=pck
            print(f'PCK: {pck}')
            info_str['pck']=pck
            # self._logger.info(OrderedDict({"PCK_key_point", pck}))
            # info_str.append(('PCK', pck))

        # if 'PCKh' in metrics:
        #     _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks, pckh_thr,
        #                                        threshold_head_box)
        #     info_str.append(('PCKh', pckh))
        #
        # if 'AUC' in metrics:
        #     info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
        #                                          auc_nor)))

        if 'EPE' in metrics:
            epe= keypoint_epe(outputs, gts, masks)
            results['EPE_key_point']= epe
            print(f'EPE: {epe}')
            info_str['epe']=epe
            # self._logger.info(OrderedDict({"EPE_key_point", epe}))

        # if 'NME' in metrics:
        #     normalize_factor = self._get_normalize_factor(
        #         gts=gts, box_sizes=box_sizes)
        #     info_str.append(
        #         ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        return info_str

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self.dataset._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.dataset.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        print(f"save results to {res_file}")
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.dataset.coco.loadRes(res_file)
        coco_eval = COCOeval(self.dataset.coco, coco_det, 'keypoints', self.sigmas, use_area=self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts


class MPIIPoseEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        config,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self.annot_root = config.dataset.kwargs.ann_file

        # for pseudo_label
        self.pseudo_labels_results = []

    def reset(self):
        self.results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        # for input, output in zip(inputs, outputs):
        self.results.append(outputs)

        #  note: sync if multi-gpu

    def evaluate(self, res_folder=None, metric='PCKh', **kwargs):
        """Evaluate PCKh for MPII dataset. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.
        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W
        Args:
            results (list[dict]): Testing results containing the following
                items:
                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['/val2017/000000\
                    397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
            res_folder (str, optional): The folder to save the testing
                results. Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.
        Returns:
            dict: PCKh for each joint
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCKh']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        kpts = []
        for result in self.results:
            preds = result['preds']
            bbox_ids = result['bbox_ids']
            batch_size = len(bbox_ids)
            for i in range(batch_size):
                kpts.append({'keypoints': preds[i], 'bbox_id': bbox_ids[i]})
        kpts = self._sort_and_unique_bboxes(kpts)

        preds = np.stack([kpt['keypoints'] for kpt in kpts]) # BxCx3

        # convert 0-based index to 1-based index,
        # and get the first two dimensions.
        preds = preds[..., :2] + 1.0  # score is removed

        if res_folder:
            pred_file = os.path.join(res_folder, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5
        gt_file = list(resources.__path__)[0] + '/mpii_gt_val.mat'
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])  # Cx3xB

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
        scaled_uv_err = uv_err / scale
        scaled_uv_err = scaled_uv_err * jnt_visible
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        PCKh = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16), dtype=np.float32)

        for r, threshold in enumerate(rng):
            less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
            pckAll[r, :] = 100. * np.sum(
                less_than_threshold, axis=1) / jnt_count

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [('Head', PCKh[head]),
                      ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
                      ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
                      ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
                      ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
                      ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
                      ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
                      ('PCKh', np.sum(PCKh * jnt_ratio)),
                      ('PCKh@0.1', np.sum(pckAll[10, :] * jnt_ratio))]
        name_value = OrderedDict(name_value)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts

def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.
    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.
        - batch_size: N
        - num_keypoints: K
    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.
    Returns:
        tuple: A tuple containing keypoint accuracy.
        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt

def keypoint_epe(pred, gt, mask):
    """Calculate the end-point error.
    Note:
        - batch_size: N
        - num_keypoints: K
    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
    Returns:
        float: Average end-point error.
    """

    distances = _calc_distances(
        pred, gt, mask,
        np.ones((pred.shape[0], pred.shape[2]), dtype=np.float32))
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))

def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.
    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)
    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size
    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T

def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.
    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.
    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1
