# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: bidirectional_lstm.py
@Software: PyCharm
@time: 2019/4/16 11:48
@desc: keras 双向lstm实现文本分类
val_acc: 0.8218
"""
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.layers import Dense,Dropout,Embedding,LSTM,Bidirectional
from keras.models import Sequential
from keras.datasets import imdb


# 去词频最高的200000个词作为特征
max_features = 200000

# 最大序列长度
maxlen = 100
bath_size = 32
print("Loading data .......")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), ':train sequences')
print(len(x_test), ':test sequences')

print('Pad sequences(Sample x time)')
x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)

print('x_train shape:', len(x_train.shape))
print('x_test shape:', len(x_test.shape))

y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=bath_size,
          epochs=5,
          validation_data=[x_test, y_test])


