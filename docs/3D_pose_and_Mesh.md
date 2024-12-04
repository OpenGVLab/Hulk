# 3D Pose Estimation and Mesh Recovery Datasets


---

## Human3.6M & coco_smpl & muco & up3d & mpii & 3dpw

1. We follow the downloading pre-processing steps of [MeshTransformer](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md) to 
download and pre-process the datasets, including `human3.6m`, `coco_smpl`, `muco`, `up3d`, `mpii`, `3dpw`.

    The `datasets` directory structure should follow the below hierarchy.
    ```
    - datasets
        - human3.6m
            - train.img.tsv
            - train.hw.tsv
            - train.linelist.tsv    
            - smpl/train.label.smpl.p1.tsv
            - smpl/train.linelist.smpl.p1.tsv
            - valid.protocol2.yaml
            - valid_protocol2/valid.img.tsv 
            - valid_protocol2/valid.hw.tsv  
            - valid_protocol2/valid.label.tsv
            - valid_protocol2/valid.linelist.tsv
        - coco_smpl  
            - train.img.tsv  
            - train.hw.tsv   
            - smpl/train.label.tsv
            - smpl/train.linelist.tsv
        - muco  
            - train.img.tsv  
            - train.hw.tsv   
            - train.label.tsv
            - train.linelist.tsv
        - up3d  
            - train.img.tsv  
            - train.hw.tsv   
            - train.label.tsv
            - train.linelist.tsv
        - mpii
            - train.img.tsv
            - train.hw.tsv
            - train.label.tsv
            - train.linelist.tsv
        - 3dpw
            - train.img.tsv
            - train.hw.tsv
            - train.label.tsv
            - train.linelist.tsv
            - test.yaml
            - test.img.tsv
            - test.hw.tsv
            - test.label.tsv
            - test.linelist.tsv
    ```

2. Transfer .tsv format datasets to the standard format.
    We provide the python script to transfer .tsv format datasets to the standard format. 
    The example of using our transfer script is as follow:
    ```
    python tsv_to_standard.py <img_file> <label_file> <hw_file> <image_dir> <pkl_file>
    ```

---

## GTA_Human
1. Please download the `GTA_Human` dataset from [here](http://caizhongang.com/projects/GTA-Human/).

2. Transfer annotations to the standard format.
    We provide the python script to transfer annotations to the standard format. 
    The example of using our transfer script is as follow:
    ```
    python process_gta_npz.py <npz_file> <anno_dir> <output_dir> 
    ```

