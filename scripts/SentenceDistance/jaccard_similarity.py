# -*- coding: utf-8 -*-
'''
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: jaccard_similarity.py
@Software: PyCharm
@time: 2019/2/21 15:11
@desc: 杰卡德系数计算：又称为 Jaccard 相似系数，用于比较有限样本集之间的相似性与差异性。Jaccard 系数值越大，样本相似度越高
       计算方式：两个样本的交集除以并集得到的数值，当两个样本完全一致时，结果为 1，当两个样本完全不同时，结果为 0
'''

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


s1 = '我不太高兴'
s2 = '我有点不高兴'
print(jaccard_similarity(s1, s2))

# 结果：0.5714285714285714