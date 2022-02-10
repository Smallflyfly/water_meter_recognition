#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 16:39

# inference
# 模型输出进行CTC对应解码，去除blank，将连续同字符合并
"""
import os

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from config.config import RecOptions
from dataset.word_dataset import WMRDataset
from model.bi_lstm import RecModelBuilder

device = torch.device("cuda")


def rec_decode(rec_prob, labelmap, blank=0):
    raw_str = torch.max(rec_prob, dim=-1)[1].data.cpu().numpy()
    res_str = []
    for b in range(len(raw_str)):
        res_b = []
        prev = -1
        for ch in raw_str[b]:
            if ch == prev or ch == blank:
                prev = ch
                continue
            res_b.append(labelmap[ch])
            prev = ch
        res_str.append(''.join(res_b))
    return res_str


def rec_load_test_image(image_path, size=(100, 32)):
    img = Image.open(image_path)
    img = img.resize(size, Image.BILINEAR)
    img = torchvision.transforms.ToTensor()(img)
    # img.sub_(0.5).div_(0.5)
    return img.unsqueeze(0)


# 测试
def rec_test():
    config = RecOptions()
    model = RecModelBuilder(rec_num_classes=config.voc_size, sDim=config.decoder_sdim)
    model.load_state_dict(torch.load(config.saved_model_path, map_location=device))
    model.eval()

    os.makedirs(config.rec_res_dir, exist_ok=True)
    _, _, labelmap = WMRDataset().gen_labelmap()  # labelmap是类别和字符对应的字典
    with torch.no_grad():
        for file in tqdm(os.listdir(config.test_dir)):
            img_path = os.path.join(config.test_dir, file)
            image = rec_load_test_image(img_path)
            batch = [image, None, None]
            pred_prob = model.forward(batch)
            print(pred_prob)
            # todo post precess
            rec_str = rec_decode(pred_prob, labelmap)[0]
            # write to file
            with open(os.path.join(config.rec_res_dir, file.replace('.jpg', '.txt')), 'w') as f:
                f.write(rec_str)


if __name__ == '__main__':
    rec_test()
