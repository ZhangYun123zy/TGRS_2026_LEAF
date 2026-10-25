# Learning to Expand and Align: A Fourier-Based Vision–Language Framework for Incremental Hyperspectral Image Classification

This is the official code of LEAF. 
Paper web page: [Learning to Expand and Align: A Fourier-Based Vision–Language Framework for Incremental Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/11739551).

## Abstract
With the rapid increase in satellite data acquisition, efficient adaptation to newly acquired hyperspectral scenes has become increasingly important. In this context, incremental learning provides a practical solution by enabling models to continuously learn new knowledge without retaining all historical data or performing full retraining. However, substantial distribution shifts across sensors, acquisition conditions, and imaging environments significantly limit the ability of existing methods to adapt from a labeled base scene to a newly acquired scene under few-shot supervision. Moreover, the cosine similarity-based prediction has difficulty capturing higher-order statistical information and complex data relationships, which are important for improving classification performance. To address these challenges, we propose a Fourier-based vision–language framework for incremental hyperspectral image classification  (HSIC), learning to expand and align (LEAF). In the base session, a Fourier-based domain expansion  strategy generates expanded domains to facilitate domain-invariant feature learning. To further enhance representation capability, a multi-level vision–language alignment  module aligns visual and textual embeddings at both instance and class levels. During the inference stage, a covariance-enhanced prediction strategy incorporates second-order statistical information to improve prediction robustness and reliability. Extensive experiments on three benchmark hyperspectral datasets demonstrate that the LEAF consistently achieves superior overall performance over state-of-the-art methods in incremental HSIC tasks.

## Citation
```
@ARTICLE{11739551,
  author={Zhang, Yun and Xi, Bobo and Li, Jiaojiao and Zhang, Hongsheng and Li, Yunsong and Xiang, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Learning to Expand and Align: A Fourier-Based Vision-Language Framework for Incremental Hyperspectral Image Classification}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2026.3744205}}
```

## Dataset
The dataset directory should look like this:
```
datasets
├── OHID
│   ├── OHID1.mat
│   ├── OHID1_gt.mat
│   ├── OHID2.mat
│   └── OHID2_gt.mat

```
## Requirement
* CUDA Version: 11.8
* PyTorch version: 2.1.2+cu118
* Python version: 3.8.19

## Usage
* For OHID dataset, you can run
`python train_all.py --dataset OHID --beta 0.01 --gama 0.1 --re_ratio 1 --training_sample_ratio 0.1 --a 0.4 --alpha 0.2 --num_epoch 100 --num_epoch_inc 350 --ensemble_alpha 0.4 --repeat_time 1`