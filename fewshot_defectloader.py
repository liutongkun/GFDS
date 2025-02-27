import numpy as np
from torch.utils.data import Dataset
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random



class FewshotDataset(Dataset):
    def __init__(self, img_dir, resize_shape,shot=1,testcategory=None):
        self.img_dir = img_dir
        self.shot = shot
        self.testcategory = testcategory
        self.resize_shape = resize_shape
        self.foldall = [
                        'tile_crack', 'tile_glue_strip', 'tile_gray_stroke', 'tile_oil', 'tile_rough',
                        'grid_bent','grid_broken','grid_glue','grid_metal_contamination','grid_thread',
                        'Steel_Am', 'Steel_Ld', 'Steel_Sc',
                        'KolektorSDD_bad',
                        'PSP_class_1','PSP_class_2','PSP_class_3','PSP_class_4','PSP_class_5','PSP_class_6','PSP_class_7',
                        'Class1_bad','Class2_bad','Class3_bad','Class4_bad','Class5_bad','Class6_bad','Class7_bad','Class8_bad',
                        'Class9_bad','Class10_bad',
                        'Small_fins', 'Small_pit', 'Small_scratch',
                        'Side_holes', 'Side_boundary',
                        'Large_bubble', 'Large_pit', 'Large_pressure', 'Large_wear',
                        'Phone_oil', 'Phone_scratch', 'Phone_stain',
                        'capsules_0','capsules_1','capsules_2','capsules_3','capsules_4',
                        'macaroni2_0','macaroni2_1','macaroni2_2','macaroni2_4','macaroni2_5','macaroni2_6'
                        ]
        self.testfold = self.testcategory
        files = glob.glob(img_dir + '**/')
        class_div_path = {}
        class_ini_num = 0
        for file in files:
            img_paths = file + 'images/'
            sub_class_paths = glob.glob(img_paths + '**/')
            for sub_class in sub_class_paths:
                imgpath = sorted(glob.glob(sub_class + '*'))
                if 'good' in sub_class:
                    gtpath = imgpath
                else:
                    gt_file = sub_class.replace('images','ground_truth')
                    gtpath = sorted(glob.glob(gt_file + '*'))
                merged_list = list(zip(imgpath, gtpath))
                sub_class_splits = sub_class.split('/')
                key_name = f'{sub_class_splits[-4]}_{sub_class_splits[-2]}'
                class_div_path[key_name] = merged_list
                class_ini_num = class_ini_num + 1

        self.all_class_num = class_ini_num
        val_path_dict = {str(key): class_div_path[str(key)] for key in self.testfold}
        self.file_path_dict_all = {str(key): class_div_path[str(key)] for key in self.foldall}
        self.file_path_dict = val_path_dict

        file_path_all_list = []
        i = 1
        self.class_index_dict = {}
        for key, value in self.file_path_dict.items():
            class_list = [i for _ in value]
            path_with_class = list(zip(value, class_list))
            file_path_all_list.extend(path_with_class)
            self.class_index_dict[i] = key
            i = i+1
        self.file_path_withclass = file_path_all_list

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.augmenters = [iaa.Fliplr(1.0),
                           iaa.Flipud(1.0),
                           iaa.Sequential([iaa.Fliplr(1.0),iaa.Flipud(1.0)]),
                           iaa.Sequential([iaa.Flipud(1.0),iaa.Fliplr(1.0), ]),
                           ]
        self.transform = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=self.mean, std=self.std)])
    def __len__(self):
        return len(self.file_path_withclass)

    def load_image(self,img_path):
        support_img = cv2.imread(img_path)
        support_img = cv2.resize(support_img, dsize=(self.resize_shape[0], self.resize_shape[1]))
        support_img = support_img.astype(np.uint8)
        support_img = Image.fromarray(cv2.cvtColor(support_img, cv2.COLOR_BGR2RGB))
        support_img = self.transform(support_img)
        return support_img

    def load_mask(self,mask_path):
        support_mask = cv2.imread(mask_path, 0)
        support_mask = cv2.resize(support_mask, dsize=(self.resize_shape[0], self.resize_shape[1]))
        support_mask = support_mask.astype(np.uint8)
        support_mask = (support_mask > 30)*1
        return support_mask

    def __getitem__(self, idx):
        query_dir, cl = self.file_path_withclass[idx]
        key = self.class_index_dict[cl]

        query_img = self.load_image(query_dir[0])
        if not 'good' in query_dir[1]:
            query_mask = self.load_mask(query_dir[1])
        else:
            query_mask = np.ones((self.resize_shape[0], self.resize_shape[1]))*255
            query_mask = query_mask.astype(np.uint8)

        support_paths = self.file_path_dict[key]
        support_paths_woquery = sorted(list(set(support_paths) - {query_dir}))
        select_num = min(self.shot,len(support_paths_woquery))
        support_paths = random.sample(support_paths_woquery,select_num)
        support_img_list = []
        support_mask_list = []
        for support_path in support_paths:
            support_img = self.load_image(support_path[0])
            support_img_list.append(support_img)
            support_mask = self.load_mask(support_path[1])
            support_mask_list.append(support_mask)

        return support_img_list, support_mask_list, query_img, query_mask, int(cl), support_paths, query_dir,key

