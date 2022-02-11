#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/11 17:04 
"""

import csv


CSV_FILE = 'sample.csv'

TXT_FILE = 'results/final_res.txt'


def create_csv():
    with open(CSV_FILE, 'wt') as csv_file:
        csv_file.write(['filename', 'result'])
        with open(TXT_FILE, 'rb') as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                line = line.strip().strip()
                name, res = line[0], line[1]
                csv_file.write([name, res])


if __name__ == '__main__':
    create_csv()