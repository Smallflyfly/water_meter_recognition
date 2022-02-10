#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 14:32 
"""
import time

import torch
from torch.utils.data import DataLoader

from config.config import RecOptions
from dataset.word_dataset import WMRDataset
import matplotlib.pyplot as plt
import numpy as np

from model.bi_lstm import RecModelBuilder
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
import tensorboardX as tb

device = torch.device("cuda")


def show_data(batch, dataset):
    image, label, label_len = batch
    image = ((image[0].permute(1, 2, 0).to('cpu').numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    plt.title('image')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()

    label_digit = label[0].to('cpu').numpy().tolist()
    label_str = ''.join([dataset.id2char[t] for t in label_digit if t > 0])

    print('label_digit: ', label_digit)
    print('label_str: ', label_str)


def train():
    config = RecOptions()
    dataset = WMRDataset(config.train_dir, max_len=config.max_len, resize_shape=(config.height, config.width),
                         train=True)
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                  shuffle=True, pin_memory=True, drop_last=False)
    batch = next(iter(train_dataloader))
    show_data(batch, dataset)
    # model
    model = RecModelBuilder(rec_num_classes=config.voc_size, sDim=config.decoder_sdim)
    model = model.to(device)
    model.train()

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=config.max_epoch)
    writer = tb.SummaryWriter()
    for epoch in range(1, config.max_epoch+1):
        for index, data in enumerate(train_dataloader):
            data = [v.to(device) for v in data]
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()

            if index % 100 == 0:
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                info = 'Time: {}, Epoch: [{}/{}] [{}/{}]'.format(t, epoch, config.max_epoch, index + 1,
                                                                 len(train_dataloader))
                print(info)
                total_loss_info = '==========>total loss: {:.6f}'.format(loss)
                print(total_loss_info)

            num_epoch = epoch * len(train_dataloader) + index
            if num_epoch % 20 == 0:
                writer.add_scalar('total_loss', loss, num_epoch)

        scheduler.step()

        if epoch % 20 == 0:
            torch.save(model.state_dict(), 'weights/word_net_{}.pth'.format(epoch))

        if epoch == config.max_epoch:
            torch.save(model.state_dict(), 'weights/word_net_last.pth')

    writer.close()


if __name__ == '__main__':
    train()
