import argparse
import os

import imageio
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import changedetection.datasets.imutils as imutils
import rasterio as rio
import re
import tifffile as tiff
import albumentations as A
import cv2

def tif_loader(path):
    with rio.open(path) as src:
        img = src.read()
        transform = src.transform
        crs = src.crs
    return dict(img=np.transpose(img, (1,2,0)), transform=transform, crs=crs)

def tiff_loader(path):
    img = tiff.imread(path)
    if img.ndim < 3:
        img = np.expand_dims(img, axis=-1)
    return img

def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    if img.shape[-1] == 4:
        img = img[:, :, :-1]
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index]+'.png')
        post_path = os.path.join(self.dataset_path, 'B', self.data_list[index]+'.png')
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index]+'.png')
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label[label > 0] = 1
        # label = label / 255 if label.max() > 2 else label

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, augment_save_path=None):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.augment_save_path = augment_save_path

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label, translate=False):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        if translate:
            pre_img, post_img = imutils.affin(pre_img, post_img, translate=0)
        
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        basename = self.data_list[index]
        if 'train' in self.data_pro_type: 
            # parts = self.data_list[index].rsplit('_', 2)

            # pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            # post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"

            pre_img_name = f"{basename}_pre_disaster.png"
            pre_mask_name = f"{basename}_pre_disaster.png"
            post_img_name = f"{basename}_post_disaster.png"
            post_mask_name = f"{basename}_post_disaster.png"
            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            loc_label_path = os.path.join(self.dataset_path, 'targets', pre_mask_name)
            clf_label_path = os.path.join(self.dataset_path, 'targets', post_mask_name)
        else:
            pre_img_name = f"{basename}_pre_disaster.png"
            post_img_name = f"{basename}_post_disaster.png"
            pre_mask_name = f"{basename}_pre_disaster_target.png"
            post_mask_name = f"{basename}_post_disaster_target.png"
            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            loc_label_path = os.path.join(self.dataset_path, 'targets', pre_mask_name)
            clf_label_path = os.path.join(self.dataset_path, 'targets', post_mask_name)

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)
        clf_label = self.loader(clf_label_path)

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label, translate=True)
            # pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            clf_label[clf_label == 0] = 255
        else:
            # pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label, translate=True)
            # pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            if self.augment_save_path is not None:
                imutils.save_affin(pre_img, os.path.join(self.augment_save_path, pre_img_name))
                imutils.save_affin(post_img, os.path.join(self.augment_save_path, post_img_name))
                
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)

class DFCDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size
        # self.colorjitter_aug = get_colorjitter_aug()
        self.use_colorjitter = False
        self.use_multiscale = False

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

            if self.use_colorjitter:
                pre_img = self.colorjitter_aug(image=pre_img)["image"]

            if self.use_multiscale:
                pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda_and_scale(pre_img, post_img, loc_label, clf_label, self.crop_size)
            else:
                pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        image_name = self.data_list[index]

        pre_img_name = f"{image_name}_pre_disaster.tif"
        post_img_name = f"{image_name}_post_disaster.tif"
        label_img_name = f"{image_name}_building_damage.tif"

        pre_path = os.path.join(self.dataset_path, 'pre-event', pre_img_name)
        post_path = os.path.join(self.dataset_path, 'post-event', post_img_name)
        label_path = os.path.join(self.dataset_path, "target", label_img_name)

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        post_img = np.stack((post_img,) * 3, axis=-1)
        assert pre_img.shape == (1024, 1024, 3)
        assert post_img.shape == (1024, 1024, 3)
        if self.data_pro_type != 'infer':
            clf_label = self.loader(label_path)
            loc_label = clf_label.copy()
            loc_label[loc_label > 0] = 1

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
        elif 'test' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
        elif 'infer' in self.data_pro_type:
            pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
            pre_img = np.transpose(pre_img, (2, 0, 1))
            post_img = imutils.normalize_img(post_img)  # imagenet normalization
            post_img = np.transpose(post_img, (2, 0, 1))
            data_idx = self.data_list[index]
            return pre_img, post_img, data_idx
        else:
            raise ValueError(f'Invalid data_pro_type, should be train, test or infer, bug found {self.data_pro_type}')

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)

def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    elif 'xBD' in args.dataset:
        dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        if torch.cuda.device_count() > 1:
            args.shuffle = False
            num_gpus = torch.cuda.device_count()
            sampler = DistributedSampler(dataset, num_replicas=num_gpus)
            # sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=8,sampler=sampler,
                                 drop_last=True)
        return data_loader
    
    elif 'SECOND' in args.dataset:
        dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    elif "bright" in args.dataset:
        dataset = DFCDamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=8,
                                 drop_last=False)
        return dataset
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
