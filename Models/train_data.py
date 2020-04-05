#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/03 14:05:47
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os

# 模型的环境
import re, pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入定义的超参hyperparameters
from Public import Config
# 引入数据预处理后的数据
import math

def get_train_val_test(features, targets):
    f_length = len(features)
    location = np.zeros([f_length, Config.LOCATION_LENTGH])
    title = np.zeros([f_length, Config.TITLE_LENGTH])
    blurb = np.zeros([f_length, Config.BLURB_LENGTH])
    for i in range(f_length):
        location[i] = np.array(features[i, 1])
        title[i] = np.array(features[i, Config.LOCATION_LENTGH])
        # features 总共有7个
        blurb[i] = np.array(features[i, 7])
    input_features = [features.take(0, 1).astype(np.float64), 
                      location, 
                      features.take(2, 1).astype(np.float64), 
                      title, 
                      features.take(4, 1).astype(np.float64), 
                      features.take(5, 1).astype(np.float64), 
                      features.take(6, 1).astype(np.float64), 
                      blurb]
    # for i in range(len(input_features)):
    #     print(input_features[i].dtype)
    #     print(type(input_features[i]))
    labels = targets
    #     分割数据集以及shuffle
    np.random.seed(100)
    number_features = len(input_features)
    shuffle_index = np.random.permutation(f_length)
    shuffle_train_index = shuffle_index[:math.ceil(f_length * 0.96)]
    shuffle_val_index = shuffle_index[math.ceil(f_length * 0.96):math.ceil(f_length * 0.98)]
    shuffle_test_index = shuffle_index[math.ceil(f_length * 0.98):]
    train_features = [input_features[i][shuffle_train_index] for i in range(number_features)]
    train_labels = labels[shuffle_train_index]
    val_features = [input_features[i][shuffle_val_index] for i in range(number_features)]
    val_lables = labels[shuffle_val_index]
    test_features = [input_features[i][shuffle_test_index] for i in range(number_features)]
    test_lables = labels[shuffle_test_index]
    return train_features, train_labels, val_features, val_lables, test_features, test_lables