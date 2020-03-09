#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os

import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(os.path.dirname(__file__)+'/data/movies-ratings.dat', sep='::', names=header,engine='python')

# 计算唯一用户和电影的数量
test=df.item_id.unique()
print(test)
n_users = np.max(df.user_id.unique())
n_items = np.max(df.item_id.unique())
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

# 将数据集分割成测试和训练。Cross_validation.train_test_split根据测试样本的比例（test_size），本例中是0.25，来将数据混洗并分割成两个数据集
train_data = df.head(250052)
test_data = df.tail(750157)
print(len(df))
# 计算数据集的稀疏度
sparsity = round(1.0 - len(df)/float(n_users*n_items), 3)
print('The sparsity level of MovieLens100K is ' + str(sparsity*100) + '%')

# 创建uesr-item矩阵，此处需创建训练和测试两个UI矩阵
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]


# 使用SVD进行矩阵分解
import scipy.sparse as sp
from scipy.sparse.linalg import svds

u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

# 利用均方根误差进行评估
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))