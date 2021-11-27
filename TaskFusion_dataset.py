# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = './MSRS/Visible/train/MSRS/'
            data_dir_ir = './MSRS/Infrared/train/MSRS/'
            data_dir_label = './MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = './MSRS/Visible/val/MSRS/'
            data_dir_ir = './MSRS/Infrared/val/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            label = np.array(Image.open(label_path))
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )
        elif self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length

# if __name__ == '__main__':
    # data_dir = '/data1/yjt/MFFusion/dataset/'
    # train_dataset = MF_dataset(data_dir, 'train', have_label=True)
    # print("the training dataset is length:{}".format(train_dataset.length))
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # train_loader.n_iter = len(train_loader)
    # for it, (image_vis, image_ir, label) in enumerate(train_loader):
    #     if it == 5:
    #         image_vis.numpy()
    #         print(image_vis.shape)
    #         image_ir.numpy()
    #         print(image_ir.shape)
    #         break
