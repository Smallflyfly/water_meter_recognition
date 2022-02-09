#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:02 
"""
from torch import nn

from loss.loss import L1BalanceCELoss
from model.decoder import SegDetector
from model.resnet import ResNet


class BasicModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = ResNet()
        self.decoder = SegDetector()

    def forward(self, data):
        output = self.backbone(data)
        output = self.decoder(output)
        return output


class SegDetectorModel(nn.Module):
    def __init__(self, device):
        super(SegDetectorModel, self).__init__()
        self.model = BasicModel()
        self.criterion = L1BalanceCELoss()
        self.device = device
        self.to(self.device)

    def forward(self, batch, training=True):
        for key, value in batch.items():
            if value is not None and hasattr(value, 'to'):
                batch[key] = value.to(self.device)

        pred = self.model(batch['image'].float())

        if self.training:
            loss, metrics = self.criterion(pred, batch)  # 计算损失函数
            return pred, loss, metrics
        else:
            return pred