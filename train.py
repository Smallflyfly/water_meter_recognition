#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:42 
"""
import time

import torch
from torch.utils.data import DataLoader

from config.config import DetOptions
from dataset.data_augment import BaseAugment, ColorJitter, RandomCropData, MakeSegDetectionData, MakeBorderMap, \
    NormalizeImage, FilterKeys
import imgaug.augmenters as iaa

from dataset.dataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np

from model.seg_detector import SegDetectorModel
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
import tensorboardX as tb

device = torch.device("cuda")


def show_image(batch):
    # 画图
    plt.figure(figsize=(60, 60))
    image = NormalizeImage.restore(batch['image'][0])
    plt.subplot(141)
    plt.title('image', fontdict={'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
    plt.subplot(142)
    plt.title('probability_map', fontdict={'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(probability_map, cmap='gray')

    threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
    plt.subplot(143)
    plt.title('threshold_map', fontdict={'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(threshold_map, cmap='gray')

    threshold_mask = (batch['thresh_mask'][0].to('cpu').numpy() * 255).astype(np.uint8)
    plt.subplot(144)
    plt.title('threshold_mask', fontdict={'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(threshold_mask, cmap='gray')

    plt.show()


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
    config = DetOptions()
    train_dataset = ImageDataset(data_dir=config.train_dir, gt_dir=config.train_gt_dir, is_training=True,
                                 processes=train_processes)
    train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=False)
    # batch = next(iter(train_dataloader))  # 获取一个 batch
    # show_image(batch)
    model = SegDetectorModel(device)
    optimizer = build_optimizer(model, optim='adam', lr=config.lr)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=config.max_epoch)
    writer = tb.SummaryWriter()
    for epoch in range(1, config.max_epoch+1):
        model.train()
        for index, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            pred, loss, metrics = model(data)
            loss.backward()
            total_loss = loss
            bce_loss, l1_loss, dice_loss = metrics['binary_loss'], metrics['thresh_loss'], metrics['thresh_binary_loss']
            optimizer.step()
            if index % 100 == 0:
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                info = 'Time: {}, Epoch: [{}/{}] [{}/{}]'.format(t, epoch, config.max_epoch, index + 1, len(train_dataloader))
                print(info)
                total_loss_info = '==========>total loss: {:.6f}'.format(total_loss)
                print(total_loss_info)
                bec_loss_info = '==========>bce loss: {:.6f}'.format(bce_loss)
                print(bec_loss_info)
                l1_loss_info = '==========>l1 loss: {:.6f}'.format(l1_loss)
                print(l1_loss_info)
                dice_loss_info = '==========>dice loss: {:.6f}'.format(dice_loss)
                print(dice_loss_info)
                print()

            num_epoch = epoch * len(train_dataloader) + index
            if num_epoch % 20 == 0:
                writer.add_scalar('total_loss', total_loss, num_epoch)
                writer.add_scalar('bce_loss', bce_loss, num_epoch)
                writer.add_scalar('l1_loss', l1_loss, num_epoch)
                writer.add_scalar('dice_loss', dice_loss, num_epoch)

        scheduler.step()

        if epoch % 40 == 0:
            torch.save(model.state_dict(), 'weights/net_{}.pth'.format(epoch))

        if epoch == config.max_epoch:
            torch.save(model.state_dict(), 'weights/last.pth')

    writer.close()


if __name__ == '__main__':
    train()