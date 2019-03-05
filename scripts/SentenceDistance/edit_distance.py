# -*- coding: utf-8 -*-
'''
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: edit_distance.py
@Software: PyCharm
@time: 2019/2/21 15:07
@desc: 编辑距离计算: 是指两个字串之间，由一个转成另一个所需的最少编辑操作次数，如果它们的距离越大，说明它们越是不同
'''

import distance

def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)

####demo1####
s1 = '两个字串之间'
s2 = '两大字串之内'
print(edit_distance(s1, s2))
# 结果：2

####demo2####
strings = [
    '你在干什么',
    '你在干啥子',
    '你在做什么',
    '你好啊',
    '我喜欢吃香蕉'
]

target = '你在干啥'
results = list(filter(lambda x: edit_distance(x, target) <= 2, strings))
print(results)
#结果：['你在干什么', '你在干啥子']