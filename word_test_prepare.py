#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 16:08 
"""
import os

import cv2
from tqdm import tqdm
import numpy as np

from config.config import DetOptions

'''
根据检测结果生成识别模型测试数据
'''


def rec_test_data_gen():
    config = DetOptions()
    test_dir = config.test_dir
    det_dir = config.det_res_dir
    word_save_dir = config.det_word_save_dir

    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(det_dir)
    for label_file in tqdm(label_files):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(det_dir, label_file), 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        line = lines[0].strip().split(',')
        locs = [float(t) for t in line[:8]]

        # image warp
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** 0.5))
        h = int(0.5 * (((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5 + ((x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'), word_image)


if __name__ == '__main__':
    rec_test_data_gen()