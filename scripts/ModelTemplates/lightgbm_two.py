# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: lightgbm_two.py
@Software: PyCharm
@time: 2019/3/7 11:52
@desc: lightgbm进行二分类
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


print("Loading Data ... ")

# 导入数据
train_x, train_y, test_x = load_data()

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
X, val_X, y, val_y = train_test_split(
    train_x,
    train_y,
    test_size=0.05,
    random_state=1,
    stratify=train_y  ## 这里保证分割后y的比例分布与原数据一致
)

X_train = X
y_train = y
X_test = val_X
y_test = val_y

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}

# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Start predicting...')

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果

# 导出结果
threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

# 导出特征重要性
importance = gbm.feature_importance()
names = gbm.feature_name()
with open('./feature_importance.txt', 'w+') as file:
    for index, im in enumerate(importance):
        string = names[index] + ', ' + str(im) + '\n'
        file.write(string)

