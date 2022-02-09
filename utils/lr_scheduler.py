#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:53 
"""

import numpy as np
import torch


class DecayLearningRate():
    def __init__(self, lr=0.004, epochs=200, factor=0.9):
        self.lr = lr
        self.epochs = epochs
        self.factor = factor

    def get_learning_rate(self, epoch):
        # 学习率随着训练过程进行不断下降
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr

    def update_learning_rate(self, optimizer, epoch):
        lr = self.get_learning_rate(epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr


def build_scheduler(optimizer, lr_scheduler='single_step', stepsize=1, gamma=0.1, max_epoch=1):
    global scheduler
    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch))

    return scheduler