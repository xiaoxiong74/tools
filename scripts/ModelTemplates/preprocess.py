# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: preprocess.py
@Software: PyCharm
@time: 2019/3/7 11:44
@desc: 通用的预处理框架
"""


import pandas as pd
import numpy as np
import scipy as sp


# 文件读取1
def read_csv_file1(f, logging=False):
    print("==========读取数据=========")
    data = pd.read_csv(f, encoding='utf-8')
    if logging:
        print(data.head(5))
        # 获取某一列并转为列表
        print(data["Question"].tolist())
        print(f, "包含以下列")
        print(data.columns.values)
        print(data.describe())
        print(data.info())
    return data


