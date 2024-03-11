import os
import time
import pickle
import random
from easydict import EasyDict as edict
import numpy as np
import torch.utils.data as data
from PIL import Image
from core.data.transforms.pedattr_transforms import PedAttrAugmentation, PedAttrTestAugmentation, PedAttrRandomAugmentation
import torch.distributed as dist


__all__ = ['AttrDataset', 'MultiAttrDataset']

def merge_pedattr_datasets(data_path_list, root_path_list, dataset_name_list, train,
                           data_use_ratio, text_label_return, select_data, ignore_other_attrs=True):
    total_img_id = []
    total_attr_num = 0
    total_img_num = 0
    total_attr_begin = []
    total_attr_end = []
    total_img_begin = []
    total_img_end = []
    total_text_dict = {}
    attr_begin = []
    attr_end = []

    for data_path, root_path, dataset_name in zip(data_path_list, root_path_list, dataset_name_list):
        assert dataset_name in ['peta', 'PA_100k', 'rap', 'rap2', 'uavhuman', 'HARDHC',
                                'ClothingAttribute', 'parse27k', 'duke', 'market','lup_0_200w', 'lup_0_600w', 'lup_600_1200w'], \
            'dataset name {} is not exist'.format(dataset_name)


        with open(data_path, 'rb') as f:
            dataset_info = pickle.load(f)
        dataset_info = edict(dataset_info)
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        if train:
            split = 'trainval'
        else:
            split = 'test'

        attr_id = dataset_info.attr_name
        attr_num = len(attr_id)

        total_attr_begin.append(total_attr_num)
        total_attr_num = total_attr_num + attr_num
        total_attr_end.append(total_attr_num)

        if select_data is None or (select_data!= None and dataset_name == select_data):
            assert split in dataset_info.partition.keys(), f'split {split} is not exist'
            img_idx = dataset_info.partition[split]

            if isinstance(img_idx, list):
                img_idx = img_idx[0]  # default partition 0

            if data_use_ratio != 1:
                img_idx = random.sample(list(img_idx), int(len(img_idx) * data_use_ratio))

            img_num = len(img_idx)
            img_idx = np.array(img_idx)

            img_id = [os.path.join(root_path, img_id[i]) for i in img_idx]
            label = attr_label[img_idx]


            total_img_begin.append(total_img_num)
            total_img_num = total_img_num + len(img_id)
            total_img_end.append(total_img_num)
        else:
            # when testing on a single dataset, split may not exist in other datasets. therefore, we need to set a fake
            # split to make the code run. and the number of images in this fake split is 0.
            # TODO: find a better way to solve this problem. e.g., use a time for-loop to load the select dataset
            img_id = []
            label = []
            img_num = 0
            total_img_begin.append(total_img_num)
            total_img_num = total_img_num + len(img_id)
            total_img_end.append(total_img_num)

    infilling_class = -1 if ignore_other_attrs else 0
    total_label = np.full((total_img_num, total_attr_num), infilling_class, dtype=np.int32)
    select_attr_begin = 0
    select_attr_end = total_attr_num

    for index, (data_path, root_path, dataset_name) in enumerate(zip(data_path_list, root_path_list, dataset_name_list)):

        assert dataset_name in ['peta', 'PA_100k', 'rap', 'rap2', 'uavhuman', 'HARDHC',
                                'ClothingAttribute', 'parse27k', 'duke', 'market','lup_0_200w', 'lup_0_600w', 'lup_600_1200w'], \
            'dataset name {} is not exist'.format(dataset_name)
        with open(data_path, 'rb') as f:
            dataset_info = pickle.load(f)
        dataset_info = edict(dataset_info)

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        if train:
            split = 'trainval'
        else:
            split = 'test'

        if not train and dataset_name != select_data:
            continue

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        attr_id = dataset_info.attr_name
        attr_num = len(attr_id)

        img_idx = dataset_info.partition[split]

        if isinstance(img_idx, list):
            img_idx = img_idx[0]  # default partition 0

        if data_use_ratio != 1:
            img_idx = random.sample(list(img_idx), int(len(img_idx) * data_use_ratio))

        img_num = len(img_idx)
        img_idx = np.array(img_idx)

        img_id = [os.path.join(root_path, img_id[i]) for i in img_idx]
        label = attr_label[img_idx]
        # import pdb;pdb.set_trace()
        if text_label_return:
            for idx in range(attr_num):
                total_text_dict[total_attr_begin[index] + idx] = eval(f"{dataset_name}_attr_name")[idx]
                
        if not train:
            if dataset_name == select_data:
                total_label[total_img_begin[index]: total_img_end[index], total_attr_begin[index]: total_attr_end[index]] = label
                total_img_id.extend(img_id)
                attr_begin.extend([total_attr_begin[index] for i in img_idx])
                attr_end.extend([total_attr_end[index] for i in img_idx])
        else:
            total_label[total_img_begin[index]: total_img_end[index], total_attr_begin[index]: total_attr_end[index]] = label
            total_img_id.extend(img_id)
            attr_begin.extend([total_attr_begin[index] for i in img_idx])
            attr_end.extend([total_attr_end[index] for i in img_idx])

    # import pdb;pdb.set_trace()
    return total_img_id, total_label, total_text_dict, attr_begin, attr_end



class MultiAttrDataset(data.Dataset):

    def __init__(self, ginfo, augmentation, task_spec, train=True, data_use_ratio=1, text_label_return=False,
                 select_data=None, ignore_other_attrs=True,
                 **kwargs):
        data_path = task_spec.data_path
        root_path = task_spec.root_path
        dataset_name = task_spec.dataset
        # import pdb; pdb.set_trace()
        self.rank = dist.get_rank()
        self.train = train
        self.img_id, self.label, self.text_dict, self.attr_begin, self.attr_end = \
            merge_pedattr_datasets(data_path, root_path, dataset_name, train,
                                   data_use_ratio, text_label_return, select_data, ignore_other_attrs)
        height = augmentation.height
        width = augmentation.width
        self.img_num = len(self.img_id)

        if train:
            self.transform = PedAttrAugmentation(height, width)
            if augmentation.get('use_random_aug', False):
                self.transform = PedAttrRandomAugmentation(height, width, \
                    augmentation.use_random_aug.m, augmentation.use_random_aug.n)
        else:
            self.transform = PedAttrTestAugmentation(height, width)


        self.task_name = ginfo.task_name

    def __getitem__(self, index):
        return self.read_one(index)

    def __len__(self):
        return len(self.img_id)

    def read_one(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.img_num)

        imgname, gt_label = self.img_id[idx], self.label[idx]
        imgpath = imgname

        try:
            img = Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            gt_label = gt_label.astype(np.float32)

            output = {}
            output = {'image': img, 'label': gt_label, 'filename': imgname, 'attr_begin': self.attr_begin[idx], 'attr_end': self.attr_end[idx]}
            
            return output
        except:
            print('{} load failed'.format(imgpath))
            return self.read_one()

    def __repr__(self):
        return self.__class__.__name__ + \
               f'rank: {self.rank} task: {self.task_name} mode:{"training" if self.train else "inference"} ' \
               f'dataset_len:{len(self.img_id)} augmentation: {self.transform}'

rap2_attr_name =  {
0: {0:'without a bald head',1:'with a bald head'},
1: {0:'with short hair',1:'with long hair'},
2: {0:'with non-black hair',1:'with black hair'},
3: {0:'without a hat',1:'with a hat'},
4: {0:'without glasses',1:'with glasses'},
5: {0:'without a shirt',1:'with a shirt'},
6: {0:'without a sweater',1:'with a sweater'},
7: {0:'without a vest',1:'with a vest'},
8: {0:'without a t-shirt',1:'with a t-shirt'},
9: {0:'without cotton',1:'with cotton'},
10: {0:'without a jacket',1:'with a jacket'},
11: {0:'without formal wear',1:'with formal wear'},
12: {0:'without tight clothes',1:'with tight clothes'},
13: {0:'without short sleeves',1:'with short sleeves'},
14: {0:'without other upper-body clothing',1:'with other upper-body clothing'},
15: {0:'without long trousers',1:'with long trousers'},
16: {0:'without a skirt',1:'with a skirt'},
17: {0:'without a short skirt',1:'with a short skirt'},
18: {0:'without a dress',1:'with a dress'},
19: {0:'without jeans',1:'with jeans'},
20: {0:'without tight trousers',1:'with tight trousers'},
21: {0:'without leather shoes',1:'with leather shoes'},
22: {0:'without sport shoes',1:'with sport shoes'},
23: {0:'without boots',1:'with boots'},
24: {0:'without cloth shoes',1:'with cloth shoes'},
25: {0:'without casual shoes',1:'with casual shoes'},
26: {0:'without other shoes',1:'with other shoes'},
27: {0:'without a backpack',1:'with a backpack'},
28: {0:'without a shoulder bag',1:'with a shoulder bag'},
29: {0:'without a handbag',1:'with a handbag'},
30: {0:'without a box',1:'with a box'},
31: {0:'without a plastic bag',1:'with a plastic bag'},
32: {0:'without a paper bag',1:'with a paper bag'},
33: {0:'without a hand trunk',1:'with a hand trunk'},
34: {0:'without other attachments',1:'with other attachments'},
35: {0:'age greater than 16',1:'age less than or equal to 16'},
36: {0:'age less than 17 or greater than 30',1:'age between 17 and 30'},
37: {0:'age less than 31 or greater than 45',1:'age between 31 and 45'},
38: {0:'age less than 46 or greater than 60',1:'age between 46 and 60'},
39: {0:'male',1:'female', 2:'gender unknown'},
40: {0:'without excess body fat',1:'with excess body fat'},
41: {0:'without normal body shape',1:'with normal body shape'},
42: {0:'without thin body shape',1:'with thin body shape'},
43: {0:'not a customer',1:'is a customer'},
44: {0:'not an employee',1:'is an employee'},
45: {0:'not calling',1:'calling'},
46: {0:'not talking',1:'talking'},
47: {0:'not gathering',1:'gathering'},
48: {0:'not holding anything',1:'holding something'},
49: {0:'not pushing anything',1:'pushing something'},
50: {0:'not pulling anything',1:'pulling something'},
51: {0:'not carrying anything in arms',1:'carrying something in arms'},
52: {0:'not carrying anything in hands',1:'carrying something in hands'},
53: {0:'no other actions',1:'performing other actions'}
}

PA_100k_attr_name =  {
0: {0:'without a hat',1:'with a hat'},
1: {0:'without glasses',1:'with glasses'},
2: {0:'without short sleeves',1:'with short sleeves'},
3: {0:'without long sleeves',1:'with long sleeves'},
4: {0:'without stripe upper-clothes',1:'with stripe upper-clothes'},
5: {0:'without logo upper-clothes',1:'with logo upper-clothes'},
6: {0:'without plaid upper-clothes',1:'with plaid upper-clothes'},
7: {0:'without splice upper-clothes',1:'with splice upper-clothes'},
8: {0:'without stripe lower-clothes',1:'with stripe lower-clothes'},
9: {0:'without pattern lower-clothes',1:'with pattern lower-clothes'},
10: {0:'without long coat',1:'with long coat'},
11: {0:'without long trousers',1:'with long trousers'},
12: {0:'without short trousers',1:'with short trousers'},
13: {0:'without skirt or dress',1:'with skirt or dress'},
14: {0:'without boots',1:'with boots'},
15: {0:'without a handbag',1:'with a handbag'},
16: {0:'without a shoulder bag',1:'with a shoulder bag'},
17: {0:'without a backpack',1:'with a backpack'},
18: {0:'not hold objects in front',1:'hold objects in front'},
19: {0:'age less than or equal to 60',1:'age greater than 60'},
20: {0:'age less than 18 or greater than 60',1:'age between 18 and 60'},
21: {0:'age greater than or equal to 18',1:'age less than 18'},
22: {0:'male',1:'female', 2:'gender unknown'},
23: {0:'not in the front position',1:'in the front position'},
24: {0:'not in the side position',1:'in the side position'},
25: {0:'not in the back position',1:'in the back position'},
}

HARDHC_attr_name =  {
0: {0:'female', 1:'male', -1:'gender unknown'},
1: {0:'with short hair',1:'with long hair'},
2: {0:'without sunglass',1:'with sunglass'},
3: {0:'without a hat',1:'with a hat'},
4: {0:'without T-skirt',1:'with T-skirt'},
5: {0:'without long sleeves',1:'with long sleeves'},
6: {0:'without formal clothes',1:'with formal clothes'},
7: {0:'without short trousers',1:'with short trousers'},
8: {0:'without jeans',1:'with jeans'},
9: {0:'without long pants',1:'with long pants'},
10: {0:'without skirt',1:'with skirt'},
11: {0:'without face mask',1:'with face mask'},
12: {0:'without logo clothes',1:'with logo clothes'},
13: {0:'without stripe clothes',1:'with stripe clothes'},
}

parse27k_attr_name =  {
0: {0:'without a bald head',1:'with a bald head'},
1: {0:'with short hair',1:'with long hair'},
2: {0:'with non-black hair',1:'with black hair'},
3: {0:'without a hat',1:'with a hat'},
4: {0:'without glasses',1:'with glasses'},
5: {0:'without a shirt',1:'with a shirt'},
6: {0:'without a sweater',1:'with a sweater'},
7: {0:'without a vest',1:'with a vest'},
8: {0:'without a t-shirt',1:'with a t-shirt'},
9: {0:'without cotton',1:'with cotton'},
10: {0:'without a jacket',1:'with a jacket'},
11: {0:'without formal wear',1:'with formal wear'},
12: {0:'without tight clothes',1:'with tight clothes'},
13: {0:'without short sleeves',1:'with short sleeves'},
14: {0:'without other upper-body clothing',1:'with other upper-body clothing'},
15: {0:'without long trousers',1:'with long trousers'},
16: {0:'without a skirt',1:'with a skirt'},
17: {0:'without a short skirt',1:'with a short skirt'},
18: {0:'without a dress',1:'with a dress'},
19: {0:'without jeans',1:'with jeans'},
20: {0:'without tight trousers',1:'with tight trousers'},
21: {0:'without leather shoes',1:'with leather shoes'},
22: {0:'without sport shoes',1:'with sport shoes'},
23: {0:'without boots',1:'with boots'},
24: {0:'without cloth shoes',1:'with cloth shoes'},
25: {0:'without casual shoes',1:'with casual shoes'},
26: {0:'without other shoes',1:'with other shoes'},
27: {0:'without a backpack',1:'with a backpack'},
28: {0:'without a shoulder bag',1:'with a shoulder bag'},
29: {0:'without a handbag',1:'with a handbag'},
30: {0:'without a box',1:'with a box'},
31: {0:'without a plastic bag',1:'with a plastic bag'},
32: {0:'without a paper bag',1:'with a paper bag'},
33: {0:'without a hand trunk',1:'with a hand trunk'},
34: {0:'without other attachments',1:'with other attachments'},
35: {0:'age greater than 16',1:'age less than or equal to 16'},
36: {0:'age less than 17 or greater than 30',1:'age between 17 and 30'},
37: {0:'age less than 31 or greater than 45',1:'age between 31 and 45'},
38: {0:'age less than 46 or greater than 60',1:'age between 46 and 60'},
39: {0:'male',1:'female', 2:'gender unknown'},
40: {0:'without excess body fat',1:'with excess body fat'},
41: {0:'without normal body shape',1:'with normal body shape'},
42: {0:'without thin body shape',1:'with thin body shape'},
43: {0:'not a customer',1:'is a customer'}
}

uavhuman_attr_name = {
0: {0:'female',1:'male'},
1: {0:'without red backpack',1:'with red backpack'},
2: {0:'without black backpack',1:'with black backpack'},
3: {0:'without green backpack',1:'with green backpack'},
4: {0:'without yellow backpack',1:'with yellow backpack'},
5: {0:'without other backpack',1:'with other backpack'},
6: {0:'without red hat',1:'with red hat'},
7: {0:'without black hat',1:'with black hat'},
8: {0:'without yellow hat',1:'with yellow hat'},
9: {0:'without white hat',1:'with white hat'},
10: {0:'without other hat',1:'with other hat'},
11: {0:'without red upper-clothes',1:'with red upper-clothes'},
12: {0:'without black upper-clothes',1:'with black upper-clothes'},
13: {0:'without blue upper-clothes',1:'with blue upper-clothes'},
14: {0:'without green upper-clothes',1:'with green upper-clothes'},
15: {0:'without multicolor upper-clothes',1:'with multicolor upper-clothes'},
16: {0:'without grey upper-clothes',1:'with grey upper-clothes'},
17: {0:'without white upper-clothes',1:'with white upper-clothes'},
18: {0:'without yellow upper-clothes',1:'with yellow upper-clothes'},
19: {0:'without dark brown upper-clothes',1:'with dark brown upper-clothes'},
20: {0:'without purple upper-clothes',1:'with purple upper-clothes'},
21: {0:'without pink upper-clothes',1:'with pink upper-clothes'},
22: {0:'without other upper-clothes',1:'with other upper-clothes'},
23: {0:'without long upper-clothes style',1:'with long upper-clothes style'},
24: {0:'without short upper-clothes style',1:'with short upper-clothes style'},
25: {0:'without skirt upper-clothes style',1:'with skirt upper-clothes style'},
26: {0:'without other upper-clothes style',1:'with other upper-clothes style'},
27: {0:'without red lower clothes',1:'with red lower clothes'},
28: {0:'without black lower clothes',1:'with black lower clothes'},
29: {0:'without blue lower clothes',1:'with blue lower clothes'},
30: {0:'without green lower clothes',1:'with green lower clothes'},
31: {0:'without multicolor lower clothes',1:'with multicolor lower clothes'},
32: {0:'without grey lower clothes',1:'with grey lower clothes'},
33: {0:'without white lower-clothes',1:'with white lower-clothes'},
34: {0:'without yellow lower-clothes',1:'with yellow lower-clothes'},
35: {0:'without dark brown lower-clothes',1:'with dark brown lower-clothes'},
36: {0:'without purple lower-clothes',1:'with purple lower-clothes'},
37: {0:'without pink lower-clothes',1:'with pink lower-clothes'},
38: {0:'without other lower-clothes',1:'with other lower-clothes'},
39: {0:'without long lower-clothes style',1:'with long lower-clothes style'},
40: {0:'without short lower-clothes style',1:'with short lower-clothes style'},
41: {0:'without skirt lower-clothes style',1:'with skirt lower-clothes style'},
42: {0:'without other lower-clothes style',1:'with other lower-clothes style'}
}

market_attr_name = {
0: {0:'without a backpack',1:'with a backpack'},
1: {0:'without a bag',1:'with a bag'},
2: {0:'without a handbag',1:'with a handbag'},
3: {0:'without black lower-clothes',1:'with black lower-clothes'},
4: {0:'without blue lower-clothes',1:'with blue lower-clothes'},
5: {0:'without brown lower-clothes',1:'with brown lower-clothes'},
6: {0:'without gray lower-clothes',1:'with gray lower-clothes'},
7: {0:'without green lower-clothes',1:'with green lower-clothes'},
8: {0:'without pink lower-clothes',1:'with pink lower-clothes'},
9: {0:'without purple lower-clothes',1:'with purple lower-clothes'},
10: {0:'without white lower-clothes',1:'with white lower-clothes'},
11: {0:'without yellow lower-clothes',1:'with yellow lower-clothes'},
12: {0:'without black upper-clothes',1:'with black upper-clothes'},
13: {0:'without blue upper-clothes',1:'with blue upper-clothes'},
14: {0:'without green upper-clothes',1:'with green upper-clothes'},
15: {0:'without gray upper-clothes',1:'with gray upper-clothes'},
16: {0:'without purple upper-clothes',1:'with purple upper-clothes'},
17: {0:'without red upper-clothes',1:'with red upper-clothes'},
18: {0:'without white upper-clothes',1:'with white upper-clothes'},
19: {0:'without yellow upper-clothes',1:'with yellow upper-clothes'},
20: {0:'with dress',1:'with pants'},
21: {0:'with long lower body clothing',1:'with short lower body clothing'},
22: {0:'with long sleeve upper body clothing',1:'with short upper body clothing'},
23: {0:'with short hair',1:'with long hair'},
24: {0:'without a hat',1:'with a hat'},
25: {0:'male',1:'female'},
26: {0:'not a young person',1:'a young person'},
27: {0:'not a teenager',1:'a teenager'},
28: {0:'not an adult',1:'an adult'},
29: {0:'not an old person',1:'an old person'}
}
# peta attr name still have some bugs
peta_attr_name = {
0: {0:'without hat accessory',1:'with hat accessory'},
1: {0:'without muffler accessory',1:'with muffler accessory'},
2: {0:'with accessory',1:'with nothing accessory'},
3: {0:'without sunglasses accessory',1:'with sunglasses accessory'},
4: {0:'with short hair',1:'with long hair'},
5: {0:'without casual upper body wear',1:'with casual upper body wear'},
6: {0:'without formal upper body wear',1:'with formal upper body wear'},
7: {0:'without jacket upper body wear',1:'with jacket upper body wear'},
8: {0:'without logo upper body wear',1:'with logo upper body wear'},
9: {0:'without plaid upper body wear',1:'with plaid upper body wear'},
10: {0:'without short sleeve upper body wear',1:'with short sleeve upper body wear'},
11: {0:'without thin stripes upper body wear',1:'with thin stripes upper body wear'},
12: {0:'without t-shirt upper body wear',1:'with t-shirt upper body wear'},
13: {0:'without other upper body wear',1:'with other upper body wear'},
14: {0:'without vneck upper body wear',1:'with vneck upper body wear'},
15: {0:'without casual lower body wear',1:'with casual lower body wear'},
16: {0:'without formal lower body wear',1:'with formal lower body wear'},
17: {0:'without jeans lower body wear',1:'with jeans lower body wear'},
18: {0:'without shorts lower body wear',1:'with shorts lower body wear'},
19: {0:'without shortskirt lower body wear',1:'with shortskirt lower body wear'},
20: {0:'without trousers lower body wear',1:'with trousers lower body wear'},
21: {0: 'without leather shoes', 1: 'with leather shoes'},
22: {0: 'without sandals', 1: 'with sandals'},
23: {0: 'without shoes', 1: 'with shoes'},
24: {0: 'without sneaker', 1: 'with sneaker'},
25: {0: 'without carrying backpack', 1: 'carrying backpack'},
26: {0: 'with carrying other things', 1: 'carrying other things'},
27: {0: 'without carrying messengerbag', 1: 'carrying messengerbag'},
28: {0: 'carrying something', 1: 'carrying nothing'},
29: {0: 'without carrying plasticbags', 1: 'carrying plasticbags'},
30: {0:'age greater than or equal to 30',1:'age less than 30'},
31: {0:'age less than 31 or greater than 45',1:'age between 31 and 45'},
32: {0:'age less than 46 or greater than 60',1:'age between 46 and 60'},
33: {0:'age less than or equal to 60',1:'age larger than 60'},
34: {0:'female',1:'male'}
}

duke_attr_name =  {
0: {0:'without a backpack',1:'with a backpack'},
1: {0:'without a bag',1:'with a bag'},
2: {0:'without boots',1:'with boots'},
3: {0:'without black lower-clothes',1:'with black lower-clothes'},
4: {0:'without blue lower-clothes',1:'with blue lower-clothes'},
5: {0:'without brown lower-clothes',1:'with brown lower-clothes'},
6: {0:'without gray lower-clothes',1:'with gray lower-clothes'},
7: {0:'without green lower-clothes',1:'with green lower-clothes'},
8: {0:'without red lower-clothes',1:'with red lower-clothes'},
9: {0:'without white lower-clothes',1:'with white lower-clothes'},
10: {0:'male',1:'female'},
11: {0:'without a handbag',1:'with a handbag'},
12: {0:'without a hat',1:'with a hat'},
13: {0:'with dark shoes',1:'with light shoes'},
14: {0:'short top clothing',1:'long top clothing'},
15: {0:'without black upper-clothes',1:'with black upper-clothes'},
16: {0:'without blue upper-clothes',1:'with blue upper-clothes'},
17: {0:'without brown upper-clothes',1:'with brown upper-clothes'},
18: {0:'without gray upper-clothes',1:'with gray upper-clothes'},
19: {0:'without green upper-clothes',1:'with green upper-clothes'},
20: {0:'without purple upper-clothes',1:'with purple upper-clothes'},
21: {0:'without red upper-clothes',1:'with red upper-clothes'},
22: {0:'without white upper-clothes',1:'with white upper-clothes'},
}

ClothingAttribute_attr_name =  ['pattern_spot', 'cyan', 'brown', 'v_shape_neckline', 'round_neckline', 'other_neckline', 'no_sleevelength', 'short_sleevelength', 'long_sleevelength', 'pattern_graphics', 'gender', 'black', 'many_colors', 'white', 'pattern_floral', 'collar', 'blue', 'necktie', 'pattern_stripe', 'pattern_solid', 'gray', 'shirt_category', 'sweater_category', 't_shirt_category', 'outerwear_category', 'suit_category', 'tank_top_category', 'dress_category', 'placket', 'pattern_plaid', 'purple', 'scarf', 'green', 'yellow', 'skin_exposure', 'red']

lup_0_200w_attr_name = {
0: {0: 'male', 1: 'female', -1:'gender unknown'},
1: {0: 'age greater than 6', 1: "age less than or equal to 6", -1: 'age unknown'},
2: {0: 'age less than 7 or greater than 18', 1: "age between 7 and 18", -1: 'age unknown'},
3: {0: 'age less than 19 or greater than 65', 1: "age between 19 and 65", -1: 'age unknown'},
4: {0: 'age less than 66', 1: "age greater than or equal to 66", -1: 'age unknown'},
5: {0: 'with short sleeve coat', 1: 'with long sleeves', -1: 'coat length unknown'},
6: {0: 'with shorts trousers', 1: 'with long trousers'},
7: {0: 'without a skirt', 1:'with a skirt'},
8: {0: 'without a pure pattern coat', 1: 'with a pure upper-clothes'},
9: {0: 'without a stripe pattern coat', 1: 'with a stripe upper-clothes'},
10: {0: 'without a design pattern coat', 1: 'with a design upper-clothes'},
11: {0: 'without a joint pattern coat', 1: 'with a joint upper-clothes'},
12: {0: 'without a lattic pattern coat', 1: 'with a lattic upper-clothes'},
13: {0: 'without a black color trousers', 1: 'with black lower-clothes'},
14: {0: 'without a white color trousers', 1: 'with white lower-clothes'},
15: {0: 'without a gray color trousers', 1: 'with a gray color trousers'},
16: {0: 'without a red color trousers', 1: 'with a red color trousers'},
17: {0: 'without a yellow color trousers', 1: 'with a yellow color trousers'},
18: {0: 'without a blue color trousers', 1: 'with a blue color trousers'},
19: {0: 'without a green color trousers', 1: 'with a green color trousers'},
20: {0: 'without a purple color trousers', 1: 'with a purple color trousers'},
21: {0: 'without a pure pattern trousers', 1: 'with a pure lower-clothes'},
22: {0: 'without a stripe pattern trousers', 1: 'with a stripe lower-clothes'},
23: {0: 'without a design pattern trousers', 1: 'with a design lower-clothes'},
24: {0: 'without a joint pattern trousers', 1: 'with a joint lower-clothes'},
25: {0: 'without a lattic pattern trousers', 1: 'with a lattic lower-clothes'},
26: {0: 'without a hat', 1: 'with a hat', -1: 'hat unknown'},
27: {0: 'without a jacket', 1: 'with a jacket'},
28: {0: 'without a sweater', 1: 'with a sweater'},
29: {0: 'without a long coat', 1: 'with a long coat'},
30: {0: 'without a shirt', 1: 'with a shirt'},
31: {0: 'without a dress', 1: 'with a dress'},
32: {0: 'without a business suit', 1: 'with a business suit'},
33: {0: 'without a black color coat', 1: 'with a black color coat', -1:'unknown coat color'},
34: {0: 'without a white color coat', 1: 'with a white color coat', -1:'unknown coat color'},
35: {0: 'without a gray color coat', 1: 'with a gray color coat', -1:'unknown coat color'},
36: {0: 'without a red color coat', 1: 'with a red color coat', -1:'unknown coat color'},
37: {0: 'without a yellow color coat', 1: 'with a yellow color coat', -1:'unknown coat color'},
38: {0: 'without a blue color coat', 1: 'with a blue color coat', -1:'unknown coat color'},
39: {0: 'without a green color coat', 1: 'with a green color coat', -1:'unknown coat color'},
40: {0: 'without a purple color coat', 1: 'with a purple color coat', -1:'unknown coat color'},
41: {0: 'with short hair', 1: 'with long hair', -1: 'unknown hair style'},
42: {0: 'without leather shoes', 1: 'with leather shoes'},
43: {0: 'without boots', 1: 'with boots'},
44: {0: 'without walking shoes', 1: 'with walking shoes'},
45: {0: 'without sandal', 1: 'with sandal'},
46: {0: 'without a bag', 1: 'without a bag', -1: 'unknown bag style'},
47: {0: 'without glasses', 1: 'with glasses'},
48: {0: 'not stand', 1: 'stand', -1: 'unknown pose'},
49: {0: 'not sit', 1: 'sit', -1: 'unknown pose'},
50: {0: 'not lie', 1: 'lie', -1: 'unknown pose'},
51: {0: 'not stoop', 1: 'stoop', -1: 'unknown pose'}}

lup_0_600w_attr_name = {
0: {0: 'male', 1: 'female', -1:'gender unknown'},
1: {0: 'age greater than 6', 1: "age less than or equal to 6", -1: 'age unknown'},
2: {0: 'age less than 7 or greater than 18', 1: "age between 7 and 18", -1: 'age unknown'},
3: {0: 'age less than 19 or greater than 65', 1: "age between 19 and 65", -1: 'age unknown'},
4: {0: 'age less than 66', 1: "age greater than or equal to 66", -1: 'age unknown'},
5: {0: 'with short sleeve coat', 1: 'with long sleeves', -1: 'coat length unknown'},
6: {0: 'with shorts trousers', 1: 'with long trousers'},
7: {0: 'without a skirt', 1:'with a skirt'},
8: {0: 'without a pure pattern coat', 1: 'with a pure upper-clothes'},
9: {0: 'without a stripe pattern coat', 1: 'with a stripe upper-clothes'},
10: {0: 'without a design pattern coat', 1: 'with a design upper-clothes'},
11: {0: 'without a joint pattern coat', 1: 'with a joint upper-clothes'},
12: {0: 'without a lattic pattern coat', 1: 'with a lattic upper-clothes'},
13: {0: 'without a black color trousers', 1: 'with black lower-clothes'},
14: {0: 'without a white color trousers', 1: 'with white lower-clothes'},
15: {0: 'without a gray color trousers', 1: 'with a gray color trousers'},
16: {0: 'without a red color trousers', 1: 'with a red color trousers'},
17: {0: 'without a yellow color trousers', 1: 'with a yellow color trousers'},
18: {0: 'without a blue color trousers', 1: 'with a blue color trousers'},
19: {0: 'without a green color trousers', 1: 'with a green color trousers'},
20: {0: 'without a purple color trousers', 1: 'with a purple color trousers'},
21: {0: 'without a pure pattern trousers', 1: 'with a pure lower-clothes'},
22: {0: 'without a stripe pattern trousers', 1: 'with a stripe lower-clothes'},
23: {0: 'without a design pattern trousers', 1: 'with a design lower-clothes'},
24: {0: 'without a joint pattern trousers', 1: 'with a joint lower-clothes'},
25: {0: 'without a lattic pattern trousers', 1: 'with a lattic lower-clothes'},
26: {0: 'without a hat', 1: 'with a hat', -1: 'hat unknown'},
27: {0: 'without a jacket', 1: 'with a jacket'},
28: {0: 'without a sweater', 1: 'with a sweater'},
29: {0: 'without a long coat', 1: 'with a long coat'},
30: {0: 'without a shirt', 1: 'with a shirt'},
31: {0: 'without a dress', 1: 'with a dress'},
32: {0: 'without a business suit', 1: 'with a business suit'},
33: {0: 'without a black color coat', 1: 'with a black color coat', -1:'unknown coat color'},
34: {0: 'without a white color coat', 1: 'with a white color coat', -1:'unknown coat color'},
35: {0: 'without a gray color coat', 1: 'with a gray color coat', -1:'unknown coat color'},
36: {0: 'without a red color coat', 1: 'with a red color coat', -1:'unknown coat color'},
37: {0: 'without a yellow color coat', 1: 'with a yellow color coat', -1:'unknown coat color'},
38: {0: 'without a blue color coat', 1: 'with a blue color coat', -1:'unknown coat color'},
39: {0: 'without a green color coat', 1: 'with a green color coat', -1:'unknown coat color'},
40: {0: 'without a purple color coat', 1: 'with a purple color coat', -1:'unknown coat color'},
41: {0: 'with short hair', 1: 'with long hair', -1: 'unknown hair style'},
42: {0: 'without leather shoes', 1: 'with leather shoes'},
43: {0: 'without boots', 1: 'with boots'},
44: {0: 'without walking shoes', 1: 'with walking shoes'},
45: {0: 'without sandal', 1: 'with sandal'},
46: {0: 'without a bag', 1: 'without a bag', -1: 'unknown bag style'},
47: {0: 'without glasses', 1: 'with glasses'},
48: {0: 'not stand', 1: 'stand', -1: 'unknown pose'},
49: {0: 'not sit', 1: 'sit', -1: 'unknown pose'},
50: {0: 'not lie', 1: 'lie', -1: 'unknown pose'},
51: {0: 'not stoop', 1: 'stoop', -1: 'unknown pose'}}

lup_600_1200w_attr_name = {
0: {0: 'male', 1: 'female', -1:'gender unknown'},
1: {0: 'age greater than 6', 1: "age less than or equal to 6", -1: 'age unknown'},
2: {0: 'age less than 7 or greater than 18', 1: "age between 7 and 18", -1: 'age unknown'},
3: {0: 'age less than 19 or greater than 65', 1: "age between 19 and 65", -1: 'age unknown'},
4: {0: 'age less than 66', 1: "age greater than or equal to 66", -1: 'age unknown'},
5: {0: 'with short sleeve coat', 1: 'with long sleeves', -1: 'coat length unknown'},
6: {0: 'with shorts trousers', 1: 'with long trousers'},
7: {0: 'without a skirt', 1:'with a skirt'},
8: {0: 'without a pure pattern coat', 1: 'with a pure upper-clothes'},
9: {0: 'without a stripe pattern coat', 1: 'with a stripe upper-clothes'},
10: {0: 'without a design pattern coat', 1: 'with a design upper-clothes'},
11: {0: 'without a joint pattern coat', 1: 'with a joint upper-clothes'},
12: {0: 'without a lattic pattern coat', 1: 'with a lattic upper-clothes'},
13: {0: 'without a black color trousers', 1: 'with black lower-clothes'},
14: {0: 'without a white color trousers', 1: 'with white lower-clothes'},
15: {0: 'without a gray color trousers', 1: 'with a gray color trousers'},
16: {0: 'without a red color trousers', 1: 'with a red color trousers'},
17: {0: 'without a yellow color trousers', 1: 'with a yellow color trousers'},
18: {0: 'without a blue color trousers', 1: 'with a blue color trousers'},
19: {0: 'without a green color trousers', 1: 'with a green color trousers'},
20: {0: 'without a purple color trousers', 1: 'with a purple color trousers'},
21: {0: 'without a pure pattern trousers', 1: 'with a pure lower-clothes'},
22: {0: 'without a stripe pattern trousers', 1: 'with a stripe lower-clothes'},
23: {0: 'without a design pattern trousers', 1: 'with a design lower-clothes'},
24: {0: 'without a joint pattern trousers', 1: 'with a joint lower-clothes'},
25: {0: 'without a lattic pattern trousers', 1: 'with a lattic lower-clothes'},
26: {0: 'without a hat', 1: 'with a hat', -1: 'hat unknown'},
27: {0: 'without a jacket', 1: 'with a jacket'},
28: {0: 'without a sweater', 1: 'with a sweater'},
29: {0: 'without a long coat', 1: 'with a long coat'},
30: {0: 'without a shirt', 1: 'with a shirt'},
31: {0: 'without a dress', 1: 'with a dress'},
32: {0: 'without a business suit', 1: 'with a business suit'},
33: {0: 'without a black color coat', 1: 'with a black color coat', -1:'unknown coat color'},
34: {0: 'without a white color coat', 1: 'with a white color coat', -1:'unknown coat color'},
35: {0: 'without a gray color coat', 1: 'with a gray color coat', -1:'unknown coat color'},
36: {0: 'without a red color coat', 1: 'with a red color coat', -1:'unknown coat color'},
37: {0: 'without a yellow color coat', 1: 'with a yellow color coat', -1:'unknown coat color'},
38: {0: 'without a blue color coat', 1: 'with a blue color coat', -1:'unknown coat color'},
39: {0: 'without a green color coat', 1: 'with a green color coat', -1:'unknown coat color'},
40: {0: 'without a purple color coat', 1: 'with a purple color coat', -1:'unknown coat color'},
41: {0: 'with short hair', 1: 'with long hair', -1: 'unknown hair style'},
42: {0: 'without leather shoes', 1: 'with leather shoes'},
43: {0: 'without boots', 1: 'with boots'},
44: {0: 'without walking shoes', 1: 'with walking shoes'},
45: {0: 'without sandal', 1: 'with sandal'},
46: {0: 'without a bag', 1: 'without a bag', -1: 'unknown bag style'},
47: {0: 'without glasses', 1: 'with glasses'},
48: {0: 'not stand', 1: 'stand', -1: 'unknown pose'},
49: {0: 'not sit', 1: 'sit', -1: 'unknown pose'},
50: {0: 'not lie', 1: 'lie', -1: 'unknown pose'},
51: {0: 'not stoop', 1: 'stoop', -1: 'unknown pose'}}
