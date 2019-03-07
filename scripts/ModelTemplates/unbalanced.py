# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: unbalanced.py
@Software: PyCharm
@time: 2019/3/7 13:43
@desc: 样本不平衡处理
"""

# 计算正负样本比例
# positive_num = df_train[df_train[’label’]==1].values.shape[0]
# negative_num = df_train[df_train[’label’]==0].values.shape[0]
# print(float(positive_num)/float(negative_num))

# 主要思路
# 手动调整正负样本比例
#
# 过采样 Over-Sampling
# 对训练集里面样本数量较少的类别（少数类）进行过采样，合成新的样本来缓解类不平衡，比如SMOTE算法
#
# 欠采样 Under-Sampling
# 将样本按比例一一组合进行训练，训练出多个弱分类器，最后进行集成
#
# 数据增强
# 如图片中可以通过旋转、剪裁、翻转等方式进行数据增强

# unbalanced处理工具https://github.com/scikit-learn-contrib/imbalanced-learn
