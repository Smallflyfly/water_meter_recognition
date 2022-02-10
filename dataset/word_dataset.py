#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 14:10 
"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class WMRDataset(Dataset):
    def __init__(self, data_dir=None, max_len=5, resize_shape=(32, 100), train=True):
        super(WMRDataset, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = train

        self.targets = [[os.path.join(data_dir, t), t.split('_')[-1][:5]] for t in os.listdir(data_dir) if
                        t.endswith('.jpg')]
        self.PADDING, self.char2id, self.id2char = self.gen_labelmap()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 可以添加更多的数据增强操作，比如 gaussian blur、shear 等
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def gen_labelmap(charset='0123456789ABCDEFGHIJ'):
        # 构造字符和数字标签对应字典
        PADDING = 'PADDING'
        char2id = {t: idx for t, idx in zip(charset, range(1, 1 + len(charset)))}
        char2id.update({PADDING: 0})
        id2char = {v: k for k, v in char2id.items()}
        return PADDING, char2id, id2char

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.targets[index][0]
        word = self.targets[index][1]
        img = Image.open(img_path)

        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        word = word[:self.max_len]
        for char in word:
            label_list.append(self.char2id[char])

        label_len = len(label_list)
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)

        if self.transform is not None and self.is_train:
            img = self.transform(img)
            img.sub_(0.5).div_(0.5)

        label_len = np.array(label_len).astype(np.int32)
        label = np.array(label).astype(np.int32)

        return img, label, label_len  # 输出图像、文本标签、标签长度, 计算 CTC loss 需要后两者信息
