#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 14:28 
"""
import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class RecBackbone(nn.Module):
    def __init__(self):
        super(RecBackbone, self).__init__()

        in_channels = 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1])  # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1])  # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1])  # [1, 25]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(_Block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_Block(self.inplanes, planes))
            return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cnn_feat = x5.squeeze(2)  # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1)

        return cnn_feat


# decoder
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn=512, nHidden=512, nOut=512):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


# basic
class RecModelBuilder(nn.Module):
    def __init__(self, rec_num_classes, sDim=512):
        super(RecModelBuilder, self).__init__()
        self.rec_num_classes = rec_num_classes
        self.sDim = sDim

        self.encoder = RecBackbone()
        self.decoder = nn.Sequential(
            BidirectionalLSTM(sDim, sDim, sDim),
            BidirectionalLSTM(sDim, sDim, rec_num_classes))

        self.rec_crit = nn.CTCLoss(zero_infinity=True)

    def forward(self, inputs):
        x, rec_targets, rec_lengths = inputs
        batch_size = x.shape[0]

        encoder_feats = self.encoder(x)  # N, T, C
        encoder_feats = encoder_feats.transpose(0, 1).contiguous()  # T, N, C
        rec_pred = self.decoder(encoder_feats)

        if self.training:
            rec_pred = rec_pred.log_softmax(dim=2)
            preds_size = torch.IntTensor([rec_pred.size(0)] * batch_size)
            loss_rec = self.rec_crit(rec_pred, rec_targets, preds_size, rec_lengths)
            return loss_rec
        else:
            rec_pred_scores = torch.softmax(rec_pred.transpose(0, 1), dim=2)
            return rec_pred_scores