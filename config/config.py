#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:28 
"""


class DetOptions():
    def __init__(self):
        self.lr = 0.004
        self.max_epoch = 200
        self.batch_size = 8
        self.num_workers = 8
        self.print_interval = 100
        self.save_interval = 10
        self.train_dir = 'data/train_imgs'
        self.train_gt_dir = 'data/train_labels/labels'
        self.test_dir = 'data/test_imgs'
        self.save_dir = 'weights/'                            # 保存检测模型
        self.saved_model_path = 'weights/net_120.pth'    # 保存最终检测模型
        self.det_res_dir = 'results/pt'                            # 保存测试集检测结
        self.det_res_vis_dir = 'results/vis'
        self.thresh = 0.3                                          # 分割后处理阈值
        self.box_thresh = 0.5                                         # 检测框阈值
        self.max_candidates = 10                                      # 候选检测框数量（本数据集每张图像只有一个文本，因此可置为1）
        self.test_img_short_side = 640                                # 测试图像最短边长度
        self.debug = False

        self.check_debug()

    def check_debug(self):
        if self.debug:
            self.max_epoch = 1
            self.print_interval = 1
            self.save_interval = 1
            self.batch_size = 2
            self.num_workers = 0