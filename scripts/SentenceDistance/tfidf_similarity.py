# -*- coding: utf-8 -*-
'''
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: tfidf_similarity.py
@Software: PyCharm
@time: 2019/2/21 15:20
@desc: 计算 TFIDF 系数：TFIDF 实际上就是在词频 TF 的基础上再加入 IDF 的信息，IDF 称为逆文档频率
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm


def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # vectors 变量实际上就对应着 TFIDF 值
    # print(vectors)
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


s1 = '我不太高兴'
s2 = '我有点不高兴'
print(tfidf_similarity(s1, s2))

# 结果：0.5803329846765686