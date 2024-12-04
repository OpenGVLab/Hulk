# 2D Human Pose Estimation Datasets

For 2D human pose estimation, we adopt the COCO format and use the datasets provided by [mmpose](https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_body_keypoint.html).

---

## COCO 
Please follow the dataset preparation steps in [mmpose](https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_body_keypoint.html#coco) to prepare the COCO datasets.

## AIC
Please follow the dataset preparation steps in [mmpose](https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_body_keypoint.html#aic) to prepare the AIC datasets.

## Human3.6m_pose
Please follow the dataset preparation steps in [mmpose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html#human3-6m) to prepare the Human3.6m_pose datasets. We provide the processed 2D pose annotation json in the [2dpose](https://drive.google.com/drive/folders/1bnSch9vJXtkff7DggIcvevoLF6Ie8PAZ?usp=drive_link).

After copying the 2D pose annotation json to the corresponding folder, the folder structure should look like:

```bash
- h36m
    - annotation_body2d
        - h36m_coco_train.json
        - h36m_coco_test.json
    - annotation_body3d
    - images
```

## posetrack
Please follow the dataset preparation steps in [mmpose](https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_body_keypoint.html#posetrack18) to prepare the posetrack datasets.

## jrdb2022
Please log in the [original website](https://jrdb.erc.monash.edu/#downloads) to download the dataset.

## MHP v2
Please download the dataset from the [official website](https://lv-mhp.github.io/dataset).

## mpi_inf_3dhp
Please download the dataset from the [official website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/).

## 3dpw
Please download the dataset from the [official website](https://virtualhumans.mpi-inf.mpg.de/3DPW/). The json file can be found in the [2dpose](https://drive.google.com/drive/folders/1bnSch9vJXtkff7DggIcvevoLF6Ie8PAZ?usp=drive_link) folder.

## AIST++
We project 3D poses in AIST++ into 2D poses. Please download the dataset from the [official website](https://google.github.io/aistplusplus_dataset/factsfigures.html). 
As it is a video dataset, we do not use all the frames for training, the training json of sampled framed cound be found in the [2dpose](https://drive.google.com/drive/folders/1bnSch9vJXtkff7DggIcvevoLF6Ie8PAZ?usp=drive_link) folder.

## Halpe (not used in Hulk, but used in UniHCP)
Please follow the dataset preparation steps in [mmpose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe) to prepare the Halpe datasets. We remove the the joints of hands and face.
The filtered json file can be found in the [2dpose](https://drive.google.com/drive/folders/1bnSch9vJXtkff7DggIcvevoLF6Ie8PAZ?usp=drive_link) folder.
