## Cycle-SNSPGAN: Towards Real-World Image Dehazing via Cycle Spectral Normalized Soft Likelihood Estimation Patch GAN (TITS'2022)

Authors: Yongzhen Wang, Xuefeng Yan, Donghai Guan, Mingqiang Wei, Yiping Chen, Xiao-Ping Zhang and Jonathan Li

[[Paper Link]](https://ieeexplore.ieee.org/document/9766195) 

### Abstract

Image dehazing is a common operation in autonomous driving, traffic monitoring and surveillance. Learning-based image dehazing has achieved excellent performance recently. However, it is nearly impossible to capture pairs of hazy/clean images from the real world to train an image dehazing network. Most of existing dehazing models that are learnt from synthetically generated hazy images generalize poorly on real-world hazy scenarios due to the obvious domain shift.
To deal with this unpaired problem arisen by real-world hazy images, we present Cycle Spectral Normalized Soft likelihood estimation Patch Generative Adversarial Network (Cycle-SNSPGAN) for image dehazing. Cycle-SNSPGAN is an unsupervised dehazing framework to boost the generalization ability on real-world hazy images.
To leverage unpaired samples of real-world hazy images without relying on their clean counterparts, we design an SN-Soft-Patch GAN and exploit a new cyclic self-perceptual loss which avoids using the ground-truth image to compute the perceptual similarity. Moreover, a significant color loss is adopted to brighten the dehazed images as human expects.
Both visual and numerical results show clear improvements of the proposed Cycle-SNSPGAN over state-of-the-arts in terms of hazy-robustness and image detail recovery, with even only a small dataset training our Cycle-SNSPGAN. Code has been available at https://github.com/yz-wang/Cycle-SNSPGAN.

#### If you find the resource useful, please cite the following :- )

```
@article{Wang_2022_TITS,
author={Wang, Yongzhen and Yan, Xuefeng and Guan, Donghai and Wei, Mingqiang and Chen, Yiping and Zhang, Xiao-Ping and Li, Jonathan},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Cycle-SNSPGAN: Towards Real-World Image Dehazing via Cycle Spectral Normalized Soft Likelihood Estimation Patch GAN}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TITS.2022.3170328}
}
```  

## Prerequisites:
Python 3.7 or above

Pytorch 1.5

CUDA 10.1

## Installation:

1. Clone this repo
2. Install PyTorch and dependencies from http://pytorch.org 

（**Note**: the code is suitable for PyTorch 1.5）



## Getting started
 

Split the dataset into train and test and place them in a folder:

```
    |─ datasets                   
    |   |─ hazy2clear        
    |   |   |─ train              # Training
    |   |   |   |─ A              # Hazy images
    |   |   |   |─ B              # Clear images
    |   |   |─ test               # Testing
    |   |   |   |─ A              # Hazy images
    |   |   |   |─ B              # Clear images
```



### Cycle-SNSPGAN Training and Test

- Train the Cycle-SNSPGAN model:
```bash
python train.py --dataroot datasets/hazy2clear/ --cuda 
```
The checkpoints will be stored at `./output`.

- Test the Cycle-SNSPGAN model:
```bash
python test.py --dataroot datasets/hazy2clear/ --cuda
```
The test results will be saved here: `./output/B`.


### Acknowledgments
Our code is developed based on [Image-Dehazing-with-GAN](https://github.com/kkjishnu/Image-Dehazing-with-GAN) and [Simpsons-Image-Colorization-using-cGAN-and-PatchGAN](https://github.com/alexandrahotti/Simpsons-Image-Colorization-using-cGAN-and-PatchGAN). We thank the awesome work provided by them.
And great thanks to the anonymous reviewers for their helpful feedback.


## Contact

If you have questions, you can contact `wangyz@nuaa.edu.cn`.

