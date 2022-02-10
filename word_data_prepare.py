#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 14:12 
"""
import os

import cv2
from tqdm import tqdm
import numpy as np

from config.config import DetOptions

'''
标签处理：定义新字符类处理半字符的情况，比如将'0-1半字符'归到'A'类，减小歧义
识别训练数据构造：从完整图像中裁剪出文本图像作为识别模型输入数据
'''


def process():
    EXT_CHARS = {
        '01': 'A', '12': 'B', '23': 'C', '34': 'D', '45': 'E',
        '56': 'F', '67': 'G', '78': 'H', '89': 'I', '09': 'J'
    }
    config = DetOptions()

    train_dir = config.train_dir
    train_labels_dir = config.train_gt_dir
    word_save_dir = config.word_train_dir  # 保存识别训练数据集
    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(train_labels_dir)
    for label_file in tqdm(label_files):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        line = lines[0].strip().split()
        locs = line[:8]
        words = line[8:]

        # 标签处理
        if len(words) == 1:
            ext_word = words[0]
        else:
            assert len(words) % 2 == 0
            ext_word = ''
            for i in range(len(words[0])):
                char_i = [word[i] for word in words]
                if len(set(char_i)) == 1:
                    ext_word += char_i[0]
                elif len(set(char_i)) == 2:
                    char_i = list(set(char_i))
                    char_i.sort()
                    char_i = ''.join(char_i)
                    ext_char_i = EXT_CHARS[char_i]
                    ext_word += ext_char_i

        locs = [int(t) for t in line[:8]]

        # 将倾斜文字图像调整为水平图像
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** 0.5))
        h = int(0.5 * (((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5 + ((x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(train_dir, label_file.replace('.txt', '.jpg')))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('.txt', '') + '_' + ext_word + '.jpg'), word_image)


if __name__ == '__main__':
    process()
