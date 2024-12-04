# Pedestrian Detection Datasets

For Pedestrian Detection, we adopt the COCO format for annotation files and filter samples without pedestrian instances. We provide the filtered json files for each dataset in the [peddet](https://drive.google.com/drive/folders/1gqeYhcQga6-h_v1vRjxb4FboVmetAJtW?usp=drive_link) folder. 

---

## CrowdHuman
Please download the dataset from the official website: [CrowdHuman](https://www.crowdhuman.org/), and put the json files into the `CrowHuman/annotations` folder.
The folder structure should be like this:
```bash
- CrowHuman
    - Images
    - annotations
        - train.json
        - val.json
    - annotation_train.odgt
    - annotation_val.odgt
```

## ECP (EuroCity Persons)
Please download the dataset from the official website: [ECP](https://eurocity-dataset.tudelft.nl/), and put the json files into the `ECP` folder.
The folder structure should be like this:
```bash
- ECP
    - day/
    - ECP_remove_no_person_img.json
```

## CityPersons
Please download the dataset from the official website: [CityPersons](https://github.com/CharlesShang/Detectron-PYTORCH/tree/master/data/citypersons), and put the json files into the `CityPersons` folder.
The folder structure should be like this:
```bash
- CityPersons
    - leftImg8bit_trainvaltest/
        - leftImg8bit/
            - train/
            - val/
            - test/
    - CityPersons_remove_no_person_img.json
```

## WiderPerson
Please download the dataset from the official website: [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/), and put the json files into the `WiderPerson` folder.
The folder structure should be like this:
```bash
- WiderPerson
    - Annotations/
    - Evaluation/
    - Images/
    - ReadMe.txt
    - WiderPerson_remove_no_person_img.json
    - test.txt
    - train.txt
    - val.txt
```

## COCO
Please download the dataset from the official website: [COCO](https://cocodataset.org/), and put the json files into the `COCO` folder.
The folder structure should be like this:
```bash
- COCO
    - train2017/
    - val2017/
    - coco_person_remove_no_person_img.json
```

## WIDER_Pedestrian
Please download the dataset from the official website: [WIDER_Pedestrian](https://competitions.codalab.org/competitions/20132), and put the json files into the `WIDER_Pedestrian` folder.
The folder structure should be like this:
```bash
- WIDER_Pedestrian
    - Images/
    - WIDER_Pedestrian_remove_no_person_img.json
```

