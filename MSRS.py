#!/usr/bin/python
# -*- encoding: utf-8 -*-


from distutils import filelist
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import os.path as osp
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from transform import *


class MSRS(Dataset):
    def __init__(
        self,
        rootpth,
        cropsize=(640, 480),
        mode='train',
        Method='SDNet',
        *args,
        **kwargs
    ):
        super(MSRS, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.filenames = []
        self.img_suffix = '.png'
        self.label_suffix = '.png'
        source_name = os.path.join(rootpth, 'split', self.mode + '.txt')
        with open(source_name) as f:
            files = f.readlines()
            for item in files:
                file_name = item.strip()
                self.filenames.append(file_name)
        ## parse img directory
        self.im_dir = os.path.join(rootpth, Method)
        self.label_dir = os.path.join(rootpth, 'Label')
        
        self.len = len(self.filenames)
        ## pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.234, 0.267, 0.231), (0.177, 0.178, 0.176)),
            ]
        )
        self.trans_train = Compose(
            [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                HorizontalFlip(),
                RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                RandomCrop(cropsize),
            ]
        )
        print('Total number of images: {}, Total number of labels: {}'.format(self.len, len(self.filenames)))

    def __getitem__(self, idx):
        img_fn = self.filenames[idx] + self.img_suffix
        label_fn = self.filenames[idx] + self.label_suffix
        impth = os.path.join(self.im_dir, img_fn)
        lbpth = os.path.join(self.label_dir, label_fn)
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label, img_fn

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


if __name__ == "__main__":
    from tqdm import tqdm

    ds = MSRS('./data/', n_classes=9, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
