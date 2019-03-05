# -*- coding: utf-8 -*-
'''
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: tf_similarity.py
@Software: PyCharm
@time: 2019/2/21 15:18
@desc:  TF 矩阵中两个向量的相似度：求解两个向量夹角的余弦值，就是点乘积除以二者的模长：cosθ=a·b/|a|*|b|
'''

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm


def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数  np.dot()向量的点乘积
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


s1 = '我不太高兴'
s2 = '我有点不高兴'
print(tf_similarity(s1, s2))

# 结果：0.7302967433402214