#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:28 
"""


class DetOptions:
    def __init__(self):
        self.lr = 0.0005
        self.max_epoch = 200
        self.batch_size = 8
        self.num_workers = 0
        self.print_interval = 100
        self.save_interval = 10
        self.train_dir = 'data/train_imgs'
        self.train_gt_dir = 'data/train_labels/labels'
        self.test_dir = 'data/test_imgs'
        self.save_dir = 'weights/'                            # 保存检测模型
        self.saved_model_path = 'weights/last.pth'    # 保存最终检测模型
        self.det_res_dir = 'results/pt'                            # 保存测试集检测结
        self.det_res_vis_dir = 'results/vis'
        self.det_word_save_dir = 'results/word_imgs'
        self.thresh = 0.3                                          # 分割后处理阈值
        self.box_thresh = 0.5                                         # 检测框阈值
        self.max_candidates = 10                                      # 候选检测框数量（本数据集每张图像只有一个文本，因此可置为1）
        self.test_img_short_side = 640                                # 测试图像最短边长度
        self.debug = False

        self.word_train_dir = 'data/word_train_imgs'

        self.check_debug()

    def check_debug(self):
        if self.debug:
            self.max_epoch = 1
            self.print_interval = 1
            self.save_interval = 1
            self.batch_size = 2
            self.num_workers = 0


class RecOptions:
    def __init__(self):
        self.height = 32  # 图像尺寸
        self.width = 100
        self.voc_size = 21  # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5  # 文本长度
        self.lr = 0.0001
        self.milestones = [40, 60]  # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 200
        self.batch_size = 32
        self.num_workers = 0
        self.print_interval = 25
        self.save_interval = 125
        self.train_dir = 'data/word_train_imgs'
        self.test_dir = 'results/word_imgs'
        self.save_dir = 'weights/'
        self.saved_model_path = 'weights/word_net_last.pth'
        self.rec_res_dir = 'results/word'

        self.debug = False

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)

    def check_debug(self):
        if self.debug:
            self.max_epoch = 1
            self.print_interval = 20
            self.save_interval = 1

            self.batch_size = 10
