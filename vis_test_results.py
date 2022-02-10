#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 10:33 
"""
import os

import cv2
from tqdm import tqdm

from config.config import DetOptions
import matplotlib.pyplot as plt
import numpy as np


def show():
    config = DetOptions()
    # 检测结果可视化
    test_dir = config.test_dir
    det_dir = config.det_res_dir
    det_vis_dir = config.det_res_vis_dir

    os.makedirs(det_vis_dir, exist_ok=True)
    label_files = os.listdir(det_dir)
    cnt = 0
    plt.figure(figsize=(60, 60))
    for label_file in tqdm(label_files):
        if not label_file.endswith('.txt'):
            continue
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))

        with open(os.path.join(det_dir, label_file), 'r') as f:
            lines = f.readlines()

        save_name = label_file.replace('det_res_', '')[:-4] + '.jpg'
        if len(lines) == 0:
            cv2.imwrite(os.path.join(det_vis_dir, save_name), image)
        else:
            line = lines[0].strip().split(',')
            locs = [float(t) for t in line[:8]]

            # draw box
            locs = np.array(locs).reshape(1, -1, 2).astype(np.int32)
            image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
            cv2.polylines(image, locs, True, (255, 255, 0), 8)

            # save images
            save_name = label_file.replace('det_res_', '')[:-4] + '.jpg'
            cv2.imwrite(os.path.join(det_vis_dir, save_name), image)

        if cnt < 4:  # 只画5张
            plt.subplot(151 + cnt)
            plt.title(save_name, fontdict={'size': 20})
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image)
            cnt += 1

    plt.show()


if __name__ == '__main__':
    show()