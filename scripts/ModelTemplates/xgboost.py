# -*- coding: utf-8 -*-
"""
@author: xiongyongfu
@contact: xyf_0704@sina.com
@file: xgboost.py
@Software: PyCharm
@time: 2019/3/7 12:01
@desc: xgboost 进行分类
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


print("Loading Data ... ")

# 导入数据
train_x, train_y, test_x = load_data()

# 构建特征


# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
X, val_X, y, val_y = train_test_split(
    train_x,
    train_y,
    test_size=0.01,
    random_state=1,
    stratify=train_y
)

# xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test_x)

# xgboost模型 #####################

params = {
    'booster': 'gbtree',
    # 'objective': 'multi:softmax',  # 多分类的问题、
    # 'objective': 'multi:softprob',   # 多分类概率
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    # 'num_class': 9,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,  # 构建树的深度，越大越容易过拟合
    'alpha': 0,  # L1正则化系数
    'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.5,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.03,  # 如同学习率
    'seed': 1000,
    'nthread': -1,  # cpu 线程数
    'missing': 1,
    'scale_pos_weight': (np.sum(y == 0) / np.sum(y == 1))
# 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)
    # 'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 2000  # 迭代次数
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

# 交叉验证
result = xgb.cv(plst, xgb_train, num_boost_round=200, nfold=4, early_stopping_rounds=200, verbose_eval=True,
                folds=StratifiedKFold(n_splits=4).split(X, y))

# 训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
model.save_model('../data/model/xgb.model')  # 用于存储训练出的模型

preds = model.predict(xgb_test)

# 导出结果
threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

