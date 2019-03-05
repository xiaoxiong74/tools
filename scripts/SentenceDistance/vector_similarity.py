# -*- coding: utf-8 -*-
'''
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: vector_similarity.py
@Software: PyCharm
@time: 2019/2/21 15:26
@desc: Word2Vec 计算：将每一个词转换为向量的过程。
'''


import gensim
import jieba
import numpy as np
from scipy.linalg import norm

model_file = '../../bin/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)


def vector_similarity(s1, s2):
    def sentence_vector(s):
        # 分词
        words = jieba.lcut(s)
        v = np.zeros(64)
        # 每一个词获取其对应的 Vector
        for word in words:
            v += model[word]
        # Vector 相加并求平均
        v /= len(words)
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    # 计算其夹角余弦值
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

s1 = '帮我查询下流量'
s2 = '查流量'
print(vector_similarity(s1, s2))
# 结果：0.9102000439587189

strings = [
    '你在干什么',
    '你在干啥子',
    '你在做什么',
    '你好啊',
    '我喜欢吃香蕉'
]

target = '你在干啥'

for string in strings:
    print(string, vector_similarity(string, target))
'''
结果：你在干什么 0.8785495016487205
     你在干啥子 0.9789649689827054
     你在做什么 0.8781992402695276
     你好啊 0.5174225914249864
     我喜欢吃香蕉 0.5829908414506211
'''