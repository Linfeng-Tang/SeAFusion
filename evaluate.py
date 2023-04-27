#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model_TII import BiSeNet
from MSRS import MSRS
# import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
from utils import SegmentationMetric

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class MscEval(object):
    def __init__(
        self,
        model,
        dataloader,
        # scales=[1],
        scales=[0.75, 0.9, 1, 1.1, 1.2, 1.25],
        # scales=[1, 1.2],
        n_classes=9,
        lb_ignore=255,
        cropsize=480,
        flip=True,
        *args,
        **kwargs
    ):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model
        print(self.scales)

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0] - H, size[1] - W
        hst, hed = margin_h // 2, margin_h // 2 + H
        wst, wed = margin_w // 2, margin_w // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def get_palette(self):
        unlabelled = [0, 0, 0]
        car = [64, 0, 128]
        person = [64, 64, 0]
        bike = [0, 128, 192]
        curve = [0, 0, 192]
        car_stop = [128, 128, 0]
        guardrail = [64, 64, 128]
        color_cone = [192, 128, 128]
        bump = [192, 64, 0]
        palette = np.array(
            [
                unlabelled,
                car,
                person,
                bike,
                curve,
                car_stop,
                guardrail,
                color_cone,
                bump,
            ]
        )
        return palette

    def visualize(self, save_name, predictions):
        palette = self.get_palette()
        # print(predictions.shape)
        # 遍历predictions
        # for (i, pred) in enumerate(predictions):
        pred = predictions
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(save_name)

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        N, C, H, W = im.size()
        long_size, short_size = (H, W) if H > W else (W, H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W - cropsize) / stride) + 1
            n_y = math.ceil((H - cropsize) / stride) + 1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = (
                        min(H, stride * iy + cropsize),
                        min(W, stride * ix + cropsize),
                    )
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        return prob

    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self, Method='NestFuse'):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        lb_ignore = [255]
        seg_metric = SegmentationMetric(n_classes, device=device)          
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank() == 0:
            dloader = self.dl
        for i, (imgs, label, fn) in enumerate(dloader):
            # if not fn[0] == '00037N.png':
            #     continue
            # print(fn[0])
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            probs_torch = torch.zeros((N, self.n_classes, H, W))
            probs_torch = probs_torch.to(device)
            probs_torch.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs_torch += prob
                probs += prob.detach().cpu()            
            seg_results = torch.argmax(probs_torch, dim=1, keepdim=True)
            seg_metric.addBatch(seg_results, label.to(device), lb_ignore)
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            for i in range(1):
                outpreds = preds[i]
                name = fn[i]
                folder_path = os.path.join('BANet', Method)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, name)
                # img = Image.fromarray(np.uint8(outpreds))
                # img.save(file_path)
                self.visualize(file_path, outpreds)
            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (
            np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist)
        )
        mIOU = np.mean(IOUs)
        mIOU = mIOU
        mIoU_torch = np.array(seg_metric.meanIntersectionOverUnion().item())
        IoU_list = IOUs.tolist()
        IoU_list.append(mIOU)
        IoU_list = [round(100 * i, 2) for i in IoU_list]
        # print('{} | IoU:{}, mIoU:{:.4f}'.format(Method, IoU_list, mIoU_torch))
        return mIOU, IoU_list


def evaluate(respth='./res_MSRS', dspth='/data/timer/BiSeNet/MSRS', Method=None, save_pth=None):
    ## logger
    logger = logging.getLogger()
    respth = os.path.join(respth, Method)
    ## model
    logger.info('\n')
    logger.info('====' * 4)
    logger.info('evaluating the model ...\n')
    # logger.info('setup and restore model')
    n_classes = 9
    net = BiSeNet(n_classes=n_classes)
    if save_pth==None:
        save_pth = osp.join(respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))
    # torch.save(net.state_dict(), save_pth, file_serialization=False)
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 1
    n_workers = 2
    dsval = MSRS(dspth, mode='test', Method=Method)
    dl = DataLoader(
        dsval,
        batch_size=batchsize,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl)

    ## eval
    mIOU, IoU_list = evaluator.evaluate(Method=Method)
    logger.info('{} | IoU:{}, mIoU:{:.4f}'.format(Method, IoU_list, mIOU))
    return mIOU


if __name__ == "__main__":
    # setup_logger('./res_Infrared')
    save_name = 'SeAFusion'
    Method = 'SeAFusion'
    respth = './model'
    log_dir = os.path.join(respth, save_name)
    os.makedirs(log_dir, exist_ok=True)
    datapth='./datasets/MSRS'
    setup_logger(log_dir)
    evaluate(respth=respth, dspth=datapth, Method=Method, save_pth='./model/Fusion/model_final.pth')