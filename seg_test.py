#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/10 10:02 
"""
import os

import torch
from tqdm import tqdm

from config.config import DetOptions
from inference.seg_detect import SegDetectorRepresenter, load_test_image, format_output
from model.seg_detector import SegDetectorModel

device = torch.device("cuda")


def test(config):
    # 模型加载
    model = SegDetectorModel(device)
    model.load_state_dict(torch.load(config.saved_model_path, map_location=device), strict=False)
    model.eval()

    # 后处理
    representer = SegDetectorRepresenter(thresh=config.thresh, box_thresh=config.box_thresh,
                                         max_candidates=config.max_candidates)

    # 推理
    os.makedirs(config.det_res_dir, exist_ok=True)
    batch = dict()
    cnt = 0
    with torch.no_grad():
        for file in tqdm(os.listdir(config.test_dir)):
            img_path = os.path.join(config.test_dir, file)
            image, ori_shape = load_test_image(img_path)
            batch['image'] = image
            batch['shape'] = [ori_shape]
            batch['filename'] = [file]
            pred = model.forward(batch, training=False)
            output = representer.represent(batch, pred)
            format_output(config.det_res_dir, batch, output)

            if config.debug and cnt >= 6:  # DEBUG
                break
            cnt += 1


if __name__ == '__main__':
    config = DetOptions()
    test(config)