<h1 align="center"><a href="https://www.sciencedirect.com/science/article/pii/S1566253521002542">Image Fusion in the Loop of High-level Vision Tasks: A Semantic-aware Real-time Infrared and Visible Image Fusion Network</a></h1>

<p align="center"><a href="https://github.com/Linfeng-Tang">Linfeng Tang</a>&emsp; Jiteng Yuan&emsp; <a href="https://sites.google.com/site/jiayima2013">Jiayi Ma</a></p>
<p align="center"><strong>Wuhan University</strong></p>
<p align="center"><strong>Information Fusion</strong> &middot; 2022</p>
<p align="center"><a href="https://esi.help.clarivate.com/Content/overview.htm"><img src="https://img.shields.io/badge/%F0%9F%94%A5_ESI_Hot-Top_0.1%25-E85D3F?style=flat-square" alt="ESI Hot Paper (top 0.1%)"></a> <a href="https://esi.help.clarivate.com/Content/overview.htm"><img src="https://img.shields.io/badge/%F0%9F%8F%86_ESI_Highly_Cited-Top_1%25-D4A017?style=flat-square" alt="ESI Highly Cited Paper (top 1%)"></a> <a href="https://www.sciencedirect.com/journal/information-fusion/about/awards/2024-inffus-best-paper-best-survey-and-best-editor-award"><img src="https://img.shields.io/badge/%F0%9F%8F%85_2024_Best_Paper_Award-Information_Fusion-7B61A8?style=flat-square" alt="Information Fusion Best Paper Award 2024"></a><br><sub><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=PyRqpAsAAAAJ&citation_for_view=PyRqpAsAAAAJ:u5HHmVD_uO8C">Google Scholar &middot; <strong>1,183 citations</strong></a> &middot; updated July 18, 2026</sub></p>

## ✨ News  
- **[2026-06-02]** Our paper **[DSPFusion: Image Fusion via Degradation and Semantic Dual-Prior Guidance](https://doi.org/10.1109/TIP.2026.3700938)** has been officially accepted by **IEEE Transactions on Image Processing (IEEE TIP)**! [[Paper](https://doi.org/10.1109/TIP.2026.3700938)] [[arXiv](https://arxiv.org/abs/2503.23355)] [[Code](https://github.com/Linfeng-Tang/DSPFusion)]
- **[2026-02-21]** Our paper **[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_VideoFusion_A_Spatio-Temporal_Collaborative_Network_for_Multi-modal_Video_Fusion_CVPR_2026_paper.html)** has been officially accepted by **The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)**! [[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_VideoFusion_A_Spatio-Temporal_Collaborative_Network_for_Multi-modal_Video_Fusion_CVPR_2026_paper.html)] [[arXiv](https://arxiv.org/abs/2503.23359)] [[Code](https://github.com/Linfeng-Tang/VideoFusion)]
- **[2025-09-18]** Our paper *[ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](https://arxiv.org/pdf/2503.23356?)* has been officially accepted by **Advances in Neural Information Processing Systems (NeurIPS 2025)**! [[Paper](https://arxiv.org/pdf/2503.23356?)] [[Code](https://github.com/Linfeng-Tang/ControlFusion)]  

- **[2025-09-10]** Our paper *[Mask-DiFuser: A Masked Diffusion Model for Unified Unsupervised Image Fusion](https://ieeexplore.ieee.org/document/11162636)* has been officially accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI)**! [[Paper](https://ieeexplore.ieee.org/document/11162636)] [[Code](https://github.com/Linfeng-Tang/Mask-DiFuser)]  

- **[2025-03-15]** Our paper *[C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning](https://github.com/Linfeng-Tang/C2RF)* has been officially accepted by the **International Journal of Computer Vision (IJCV)**! [[Paper](https://link.springer.com/article/10.1007/s11263-025-02427-1)] [[Code](https://github.com/Linfeng-Tang/C2RF)]  

- **[2025-02-11]** We released a large-scale dataset for infrared and visible video fusion: *[M2VD: Multi-modal Multi-scene Video Dataset](https://github.com/Linfeng-Tang/M2VD)*.  

- **[2024-11-28]** *SeAFusion* won the **Information Fusion Best Paper Award 2024**! 

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
@article{Tang2024Mask-DiFuser,
  author={Tang, Linfeng and Li, Chunyu and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Mask-DiFuser: A Masked Diffusion Model for Unified Unsupervised Image Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
 }
```

```
@article{Tang2024C2RF,
	title={C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning}, 
	author={Tang, Linfeng and Yan, Qinglong and Xiang, Xinyu and Fang, Leyuan and Ma, Jiayi},
	journal={International Journal of Computer Vision}, 
	pages={5262--5280},
	volume={133},
	year={2025},
}
```
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
