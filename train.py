#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:42 
"""
from torch.utils.data import DataLoader

from config.config import DetOptions
from dataset.data_augment import BaseAugment, ColorJitter, RandomCropData, MakeSegDetectionData, MakeBorderMap, \
    NormalizeImage, FilterKeys
import imgaug.augmenters as iaa

from dataset.dataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np


def train():
    train_processes = [
        BaseAugment(only_resize=False, keep_ratio=False,
                    augmenters=iaa.Sequential([
                        iaa.Fliplr(0.5),  # 水平翻转
                        iaa.Affine(rotate=(-10, 10)),  # 旋转
                        iaa.Resize((0.5, 3.0))  # 尺寸调整
                    ])),
        ColorJitter(),  # 颜色增强
        RandomCropData(size=[640, 640]),  # 随机裁剪
        MakeSegDetectionData(),  # 构造 probability map
        MakeBorderMap(),  # 构造 threshold map
        NormalizeImage(),  # 归一化
        FilterKeys(superfluous=['polygons', 'filename', 'shape', 'ignore_tags', 'is_training']),  # 过滤多余元素
    ]
    det_args = DetOptions()
    train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir, is_training=True,
                                 processes=train_processes)
    train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=False)
    batch = next(iter(train_dataloader))  # 获取一个 batch

    print(batch)

    # 画图
    plt.figure(figsize=(60, 60))
    image = NormalizeImage.restore(batch['image'][0])
    plt.subplot(141)
    plt.title('image', fontdict={'size': 60})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
    plt.subplot(142)
    plt.title('probability_map', fontdict={'size': 60})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(probability_map, cmap='gray')

    threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
    plt.subplot(143)
    plt.title('threshold_map', fontdict={'size': 60})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(threshold_map, cmap='gray')

    threshold_mask = (batch['thresh_mask'][0].to('cpu').numpy() * 255).astype(np.uint8)
    plt.subplot(144)
    plt.title('threshold_mask', fontdict={'size': 60})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(threshold_mask, cmap='gray')


if __name__ == '__main__':
    train()