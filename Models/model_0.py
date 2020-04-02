#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 15:45:40
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
# 模型的环境
import re, pickle
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
import time
from tensorflow import keras
# for module in tf, np, pd, sklearn, tf, keras:
#     print(module.__name__, module.__version__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入定义的超参hyperparameters
from Public import hyperparameters as hp
from Public import Config
# 引入我们构建神经网络所需的层
try:
    # 当外部调用同目录包的时候需要使用相对路径访问
    from .Base_layers import get_inputs, get_user_embedding, get_user_feature_gru, get_book_embedding, get_book_feature_gru, get_rating
except ModuleNotFoundError:
    # 当本地运行测试的时候，需要使用直接from import，不然会报错ModuleNotFoundError: No module named '__main__.xxxxxx'; '__main__' is not a package
    from Base_layers import get_inputs, get_user_embedding, get_user_feature_gru, get_book_embedding, get_book_feature_gru, get_rating
# 引入数据预处理后的数据
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

# 定义好我们所需要的所有features和targets
features = origin_DATA.features.values
targets=origin_DATA.labels.values

"""
模型0
"""


import math
f_length = len(features)

def get_train_val_test():
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

train_features, train_labels, val_features, val_lables, test_features, test_lables = get_train_val_test()

class Net_works(object):
    def __init__(self, 
        batch_size=256,
        epoch=5):
        self.batchsize = batch_size
        self.epoch = epoch
         # 获取输入占位符
        user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb = get_inputs()
        # 获取User的2个嵌入向量
        uid_embed_layer, location_embed_layer = get_user_embedding(user_id,user_location)
        # 得到用户特征
        user_dense_layer,user_dense_layer_flat =get_user_feature_gru(uid_embed_layer,location_embed_layer)
        # 获取书籍的嵌入向量
        book_isbn_embed_layer,book_author_embed_layer,book_year_embed_layer,book_publisher_embed_layer,book_title_embed_layer,book_blurb_embed_layer=get_book_embedding(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb)
        # 获取书籍特征
        book_dense_layer,book_dense_layer_flat=get_book_feature_gru(book_isbn_embed_layer,book_author_embed_layer,book_year_embed_layer,book_publisher_embed_layer,book_title_embed_layer,book_blurb_embed_layer)
        
        # 计算出评分
        # 将用户特征和电影特征做矩阵乘法得到一个预测评分的方案
        print("user_dense_layer_flat=",user_dense_layer_flat.shape)
        print("book_dense_layer_flat=",book_dense_layer_flat.shape)
        inference = get_rating(user_dense_layer_flat,book_dense_layer_flat)

        self.model = tf.keras.Model(
            inputs=[user_id, user_location, book_isbn, book_title, book_author, book_year, book_publisher, book_blurb],
            outputs=[inference])
        self.model.summary()

    def train_model(self):
        model_optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=model_optimizer, loss=keras.losses.mse)
        history = self.model.fit(train_features, train_labels, validation_data=(val_features, val_lables), epochs=self.epoch, batch_size=self.batchsize, verbose=1)
        return  history

    def predict_model(self, model):
        test_loss = self.model.evaluate(test_features, test_lables, verbose=0)
        return test_loss

if __name__ == "__main__":
    pass
else:
    pass