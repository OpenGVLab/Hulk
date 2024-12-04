# Human Parsing Datasets

---

## Human3.6m Parsing

For Human3.6M, please download from the [official website](http://vision.imar.ro/human3.6m/description.php) and run the [preprocessing script](https://github.com/hongfz16/HCMoCo) from HCMOCO. The folder structure should look like:


```bash
- human3.6m
    - protocol_1/
        - rgb
        - seg
    - flist_2hz_train.txt
    - flist_2hz_eval.txt
```

## LIP & CIHP & ATR

For LIP & CIHP & ATR, please download from the [official website](http://sysu-hcp.net/lip/index.php) (Section 2.2 Multi-Person). The train_id.txt and val_id.txt can be found from the [parsing](https://drive.google.com/drive/folders/17wTMAFx62TAXP0gIM-z3GRCiyuHxX6O4?usp=drive_link) folder.

After copying the train_idx.txt,

LIP folder structure should look like:

```bash
- LIP
    -data
        -train_id.txt
        -train_images
            -1000_1234574.jpg
        -val_images
        -val_id.txt
    -Trainval_parsing_annotations
        -train_segmentations
            -1000_1234574.png
```

CIHP folder structure should look like:

```bash
- CIHP
    -instance-level_human_parsing
        -Training
            -Images
                -0008522.jpg
            -Category_ids
                -0008522.png
            -train_id.txt
        -Validation
            -val_id.txt
```

ATR folder structure should look like:

```bash
- ATR
    -humanparsing
        -JPEGImages
        -SegmentationClassAug
    -train_id.txt
    -val_id.txt
```

## VIP
For VIP, please download VIP_Fine from the [official website](https://sysu-hcp.net/lip/overview.php) (Section 2.3 Video Multi-Person Human Parsing). We only use it for training, and the train_id.txt can be found from the [parsing](https://drive.google.com/drive/folders/17wTMAFx62TAXP0gIM-z3GRCiyuHxX6O4?usp=drive_link) folder.
After copying the train_idx.txt, the folder structure should look like:
```bash
- VIP
    - Images/
    - Annotations/
        - Category_ids/
            - videos1/
                - 000000000001.png
        - ...
    - train_id.txt
```

## ModaNet
For ModaNet, please download from the [official website](https://github.com/eBay/modanet). We only use it for training, and the train_id.txt can be found from the [parsing](https://drive.google.com/drive/folders/17wTMAFx62TAXP0gIM-z3GRCiyuHxX6O4?usp=drive_link) folder.
After copying the train_idx.txt, the folder structure should look like:
```bash
- ModaNet
    - images/
    - annotations/
    - train_id.txt
```

## DeepFashion2
For DeepFashion2, please download from the [official website](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok) for train.zip and validation.zip. We only use it for training, and the train_id.txt can be found from the [parsing](https://drive.google.com/drive/folders/17wTMAFx62TAXP0gIM-z3GRCiyuHxX6O4?usp=drive_link) folder.
After copying the train_idx.txt, the folder structure should look like:
```bash
- DeepFashion2
    - train/
    - validation/
    - train_id.txt
```

 