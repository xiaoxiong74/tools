# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: keras_two.py
@Software: PyCharm
@time: 2019/3/7 12:02
@desc: keras 进行二分类
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

imp = Imputer(missing_values='NaN', strategy ='mean', axis = 0)
X_train = imp.fit_transform(X_train)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

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
model.add(Dense(1))  # 这里需要和输出的维度一致
model.add(Activation('sigmoid'))

# For a multi-class classification problem
model.compile(loss='binary_crossentropy',
optimizer ='rmsprop',
metrics = ['accuracy'])

epochs = 100
model.fit(X_train, y, epochs=epochs, batch_size=2000, validation_split=0.1, shuffle=True)

# 导出结果
threshold = 0.5
for index, case in enumerate(X_test):
    case = np.array([case])
    prediction_prob = model.predict(case)
    prediction = 1 if prediction_prob[0][0] > threshold else 0
