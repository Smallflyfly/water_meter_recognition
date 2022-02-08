#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/8 15:50 

BCE loss 计算 probability map 损失

MaskL1Loss 计算 threshold map 损失

DiceLoss 计算 approximate binary map 损失 DiceLoss 介绍

"""
import torch
from torch import nn


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss


# MaskL1Loss 计算 threshold map 损失
class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss


# DiceLoss 计算 approximate binary map 损失
class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape

        interp = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * interp / union
        return loss


class L1BalanceCELoss(nn.Module):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=1):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale  # 不同损失赋予不同权重
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        metrics = dict()
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        l1_loss = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])

        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics['binary_loss'] = bce_loss
        metrics['thresh_loss'] = l1_loss
        metrics['thresh_binary_loss'] = dice_loss

        return loss, metrics