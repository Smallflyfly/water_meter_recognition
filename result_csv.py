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
    with open(CSV_FILE, 'wt', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'result'])
        ls = []
        with open(TXT_FILE, 'r') as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                line = line.strip().split()
                name, res = str(line[0]), str(line[1])
                ls.append([name, res])
        csv_writer.writerows(ls)


if __name__ == '__main__':
    create_csv()