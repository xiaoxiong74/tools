# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: keras_mult.py
@Software: PyCharm
@time: 2019/3/7 13:39
@desc: keras 进行多分类
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import StandardScaler  # 用于特征的标准化
from sklearn.preprocessing import Imputer

print("LoadingData … ")
# 导入数据
train_x, train_y, test_x = load_data()

# 构建特征
X_train = train_x.values
X_test = test_x.values
y = train_y

# 特征处理
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
y = to_categorical(y)  ## 这一步很重要，一定要将多类别的标签进行one-hot编码

model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],)))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('linear'))
model.add(Dense(9))  # 这里需要和输出的维度一致
model.add(Activation('softmax'))

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
loss ='categorical_crossentropy',
metrics = ['accuracy'])

epochs = 200
model.fit(X_train, y, epochs=epochs, batch_size=200, validation_split=0.1, shuffle=True)

# 导出结果
for index, case in enumerate(X_test):
    case = np.array([case])
    prediction_prob = model.predict(case)
    prediction = np.argmax(prediction_prob)
