

#  SeAFusion

This is official Pytorch implementation of "[Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network](https://www.sciencedirect.com/science/article/pii/S1566253521002542)"
## Welcome to follow the further work of our SeAFusion：[Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity](https://github.com/Linfeng-Tang/PSFusion) 【[Paper](https://www.sciencedirect.com/science/article/pii/S1566253523001860)】, 【[Code](https://github.com/Linfeng-Tang/PSFusion)】.
## Framework
![The overall framework of the proposed semantic-aware infrared and visible image fusion algorithm.](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Figure/Framework.png)
The overall framework of the proposed semantic-aware infrared and visible image fusion algorithm.

## Network Architecture
![The architecture of the real-time infrared and visible image fusion network based on gradient residual dense block.](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Figure/Network.png)
The architecture of the real-time infrared and visible image fusion network based on gradient residual dense block.

## To Train

Run ```**CUDA_VISIBLE_DEVICES=0 python train.py**``` to train your model.
The training data are selected from the MFNet dataset. For convenient training, users can download the training dataset from [here](https://pan.baidu.com/s/1xueuKYvYp7uPObzvywdgyA), in which the extraction code is: **bvfl**.

The MFNet dataset can be downloaded via the following link: [https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw](https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw).

The MFNet project address is: [https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/).
## To Test

Run ```**CUDA_VISIBLE_DEVICES=0 python test.py**``` to test the model.

## For quantitative evaluation
For quantitative assessments, please follow the instruction to modify and run **. /Evaluation/test_evaluation.m** .

## Recommended Environment

 - [ ] torch  1.7.1
 - [ ] torchvision 0.8.2
 - [ ] numpy 1.19.2
 - [ ] pillow  8.0.1

## Fusion Example
![Qualitative comparison of SeAFusion with 9 state-of-the-art methods on 00633D image from the MFNet dataset.](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Figure/00633D.png)
Qualitative comparison of SeAFusion with 9 state-of-the-art methods on 00633D image from the MFNet dataset.

## Segmentation Results
![Segmentation results for infrared, visible and fused images from the MFNet dataset.](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Figure/Segmentation1.png)
Segmentation results for infrared, visible and fused images from the MFNet dataset. The segmentation models are re-trained on infrared, visible and fused image sets.
Each two rows represent a scene.

![Segmentation results for infrared, visible and fused images from the MFNet dataset.](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Figure/Segmentation_Deeplab.png)
Segmentation results for infrared, visible and fused images from the MFNet dataset. The segmentation model is Deeplabv3+, pre-trained on the Cityscapes dataset. Each
two rows represent a scene.

## Detection Results
![Object detection results for infrared, visible and fused images from the MFNet dataset.](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Figure/Detection.png)
Object detection results for infrared, visible and fused images from the MFNet dataset. The YOLOv5 detector, pre-trained on the Coco dataset is deployed to achieve
object detection.


## If this work is helpful to you, please cite it as：
```
@article{TANG202228SeAFusion,
title = {Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
journal = {Information Fusion},
volume = {82},
pages = {28-42},
year = {2022},
issn = {1566-2535}
}
```
