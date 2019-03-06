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

# 这里的limit 是取词频最高的100000个词，不要则表示加载全部词向量
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True, limit=100000)


def vector_similarity(s1, s2):
    def sentence_vector(s):
        # 分词
        words = jieba.lcut(s)
        v = np.zeros(64)
        # 每一个词获取其对应的 Vector
        for word in words:
            # 未登录词处理(即输入的词不在model.vocab中则忽略掉)
            if word in model.vocab:
                v += model[word]
        # Vector 相加并求平均
        v /= len(words)
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    # 计算其夹角余弦值
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

# s1 = '帮我查询下流量'
# s2 = '查流量'
# print(vector_similarity(s1, s2))
# 结果：0.9102000439587189


strings = [
    '查询渝北张三的话费',
    '查流量',
    '查下重庆的天气',
    '查询重庆的抢劫案',
    '查冉家坝的案子'
]

target = '查询渝北的杀人案子'

for string in strings:
    print(string, vector_similarity(string, target))
'''
结果：
查询渝北张三的话费 0.7075828565367552
查流量 0.39603366589562944
查下重庆的天气 0.4157257718381809
查询重庆的抢劫案 0.8367542312365408
查冉家坝的案子 0.7903167412307478
'''