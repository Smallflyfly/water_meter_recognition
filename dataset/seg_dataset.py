#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:26 
"""
import os

import cv2
from torch.utils.data import Dataset
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, data_dir=None, gt_dir=None, is_training=True, processes=None):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.is_training = is_training
        self.processes = processes

        self.image_paths = []
        self.gt_paths = []

        image_list = os.listdir(self.data_dir)
        self.image_paths = [self.data_dir + '/' + t for t in image_list]
        self.gt_paths = [self.gt_dir + '/' + t.replace('.jpg', '.txt') for t in image_list]
        self.targets = self.load_ann()  # 导入标注信息

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                line = line.strip().split()
                poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = line[8]  # 前8为 polygon 坐标，第9是文本字符串
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index):
        if index >= len(self.image_paths):
            index = index % len(self.image_paths)
        data = {}
        image_path = self.image_paths[index]

        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        target = self.targets[index]

        data['filename'] = image_path.split('/')[-1]
        data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        data['lines'] = target
        data['is_training'] = self.is_training
        if self.processes is not None:
            for data_process in self.processes:  # 做数据增强
                data = data_process.process(data)
        return data

    def __len__(self):
        return len(self.image_paths)