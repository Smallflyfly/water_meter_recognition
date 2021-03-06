#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/11 16:38 
"""
import os

from config.config import RecOptions

'''
识别结果后处理
'''


def final_postProcess():
    SPECIAL_CHARS = {k: v for k, v in zip('ABCDEFGHIJ', '1234567890')}

    config = RecOptions()
    test_dir = config.test_dir
    rec_res_dir = config.rec_res_dir
    rec_res_files = os.listdir(rec_res_dir)

    final_res = dict()
    with open('results/final_res.txt', 'w', encoding='utf-8') as f1:
        for index in range(1, 501):
            file = 'test_{}.jpg'.format(index)
            res_file = file.replace('.jpg', '.txt')
            if res_file not in rec_res_files:
                final_res[file] = ''
                continue
            with open(os.path.join(rec_res_dir, res_file), 'r') as f2:
                rec_res = f2.readline().strip()
                rec_res = ''.join([t if t not in 'ABCDEFGHIJ' else SPECIAL_CHARS[t] for t in rec_res])
                f1.write(file + ' ' + rec_res + '\n')
            # final_res[file] = ''.join([t if t not in 'ABCDEFGHIJ' else SPECIAL_CHARS[t] for t in rec_res])

    # with open('results/final_res.txt', 'w') as f:
    #     for key, value in final_res.items():
    #         f.write(key + '\t' + value + '\n')


if __name__ == '__main__':
    # 生成最终的测试结果
    final_postProcess()