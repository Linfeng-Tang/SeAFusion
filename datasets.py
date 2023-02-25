# coding:utf-8
import torchvision.transforms.functional as TF
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import glob
from natsort import natsorted
class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            self.vis_dir = './MSRS/Visible/train/MSRS/'
            self.ir_dir = './MSRS/Infrared/train/MSRS/'
            self.label_dir = './MSRS/Label/train/MSRS/'
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif split == 'val' or split == 'test':
            self.vis_dir = vi_path
            self.ir_dir = ir_path
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

    def __getitem__(self, index):
        img_name = self.filelist[index]
        vis_path = os.path.join(self.vis_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)          
        img_vis = self.imread(path=vis_path)
        img_ir = self.imread(path=ir_path, vis_flage=False)            
        if self.split=='train':            
            label_path = os.path.join(self.label_dir, img_name)  
            label = self.imread(path=label_path, label=True)
            label = label.type(torch.LongTensor)   
                  
        if self.split=='train': 
            return img_vis, img_ir, label, img_name
        else:
            return img_vis, img_ir, img_name

    def __len__(self):
        return self.length
    
    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img) * 255
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img)
            else: ## infrared images single channel 
                img = Image.open(path).convert('L') 
                im_ts = TF.to_tensor(img)
        return im_ts
