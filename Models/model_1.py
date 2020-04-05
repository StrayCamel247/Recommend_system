#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 22:16:56
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys, os
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
    from .train_data import get_train_val_test
    from .Base_layers import get_inputs, get_rating
except ModuleNotFoundError:
    # 当本地运行测试的时候，需要使用直接from import，不然会报错ModuleNotFoundError: No module named '__main__.xxxxxx'; '__main__' is not a package
    from train_data import get_train_val_test
    from Base_layers import get_inputs, get_rating
    
# 引入数据预处理后的数据
from Data import origin_DATA, all_user_id_number, all_location_words_number, all_isbn_words_number, all_title_words_number, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

# 构建User神经网络
def get_user_embedding_16(u_id, u_loca):
    user_id_embedd = keras.layers.Embedding(all_user_id_number, hp.embedding_dim//2, name='user_id_embedding')(u_id)
    user_loca_embedd = keras.layers.Embedding(all_location_words_number, hp.embedding_dim//2 , name='user_loca_embedding')(u_loca)
    return user_id_embedd, user_loca_embedd


def get_user_feature_lstm(u_id_embedd, u_loca_embedd):
    """
    model_0: 使用lstm处理用户信息中的location特征
    """
    u_id_layer = keras.layers.Dense(64, activation='relu', name='u_id_dense')(u_id_embedd)
    # u_id_layer.shape = (?, 1, 64)
    # u_loca_layer.shape = (?, 64)
    # 这里可以再加个Dense
    u_loca_layer = keras.layers.LSTM(32, go_backwards=False, name='u_loca_lstm')(u_loca_embedd)
    u_loca_layer_lstm = keras.layers.Dense(64, activation='relu', name='u_loca_layer_lstm')(u_loca_layer)
    u_id_reshape = keras.layers.Reshape([64])(u_id_layer)
    u_combine = keras.layers.concatenate([u_id_reshape, u_loca_layer_lstm],axis=1, name='u_combine')
    print(u_combine.shape)
    # 这里能不能用激活函数
    u_feature_layer = keras.layers.Dense(200, name='u_feature_layer')(u_combine)
    print(u_feature_layer.shape)
    return u_feature_layer


# 构建Book神经网络
def get_book_embedding_16(b_isbn, b_atuhor, b_year, b_publisher, b_title, b_blurb):
    book_isbn_embedd = keras.layers.Embedding(all_isbn_words_number, hp.embedding_dim//2, name='book_isbn_embedding')(b_isbn)
    book_author_embedd = keras.layers.Embedding(all_author_words_number, hp.embedding_dim//2, name='book_author_embedding')(b_atuhor)
    book_year_embedd = keras.layers.Embedding(all_year_words_number, hp.embedding_dim//2, name='book_year_embedding')(b_year)
    book_publisher_embedd = keras.layers.Embedding(all_publisher_words_number, hp.embedding_dim//2, name='book_publisher_embedding')(b_publisher)
    
    book_title_embedd = keras.layers.Embedding(all_title_words_number, hp.embedding_dim, name='book_title_embedding')(b_title)
    book_blurb_embedd = keras.layers.Embedding(all_blurb_words_number, hp.embedding_dim, name='book_blurb_embedding')(b_blurb)
    return book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd


def get_book_feature_lstm(b_isbn_embedd, b_author_embedd, b_year_embedd, b_publisher_embedd, b_title_embedd, b_blurb_embedd):
    """
    model_1: 使用lstm处理书籍信息中的blurb特征
    """
    # 首先对前4个特征连接Dense层
    b_isbn_dense = keras.layers.Dense(hp.dense_dim//2, activation='relu', name='b_isbn_dense')(b_isbn_embedd)
    b_author_dense = keras.layers.Dense(hp.dense_dim//2, activation='relu', name='b_author_dense')(b_author_embedd)
    b_year_dense = keras.layers.Dense(hp.dense_dim//2, activation='relu', name='b_year_dense')(b_year_embedd)
    b_publisher_dense = keras.layers.Dense(hp.dense_dim//2, activation='relu', name='b_publisher_dense')(b_publisher_embedd)
    # 合并这四个特征,  b_combine_four shape = (?, 1, 64)
    b_combine_four = keras.layers.concatenate([b_isbn_dense, b_author_dense, b_year_dense, b_publisher_dense], name='b_four_combine')
    
    # 对title进行卷积
    b_title_reshape = keras.layers.Lambda(lambda layer: tf.expand_dims(layer, 3))(b_title_embedd)  # shape=(?,15, 32, 1)
    print('b_title_reshape.shape = ', b_title_reshape.shape)
    b_title_conv = keras.layers.Conv2D(filters=8, kernel_size=(2, hp.dense_dim//2), strides=1)(b_title_reshape)# shape=(?, 14, 1, 8)
    b_title_pool = keras.layers.MaxPool2D(pool_size=(14, 1), strides=1)(b_title_conv) # shape=(?,1, 1, 8)
    
    # 对blurb进行处理
    b_blurb_lstm = keras.layers.LSTM(32, name='b_blurb_lstm')(b_blurb_embedd) # shape = (?, 32)
    # 将title和blurb合并
    b_title_reshape = keras.layers.Reshape([b_title_pool.shape[3]])(b_title_pool)
    # b_combine_blurb_title.shape = (?, 40)
    b_combine_blurb_title = keras.layers.concatenate([b_title_reshape, b_blurb_lstm], axis=1, name='b_combine_blurb_title')
    b_blurb_title_dense = keras.layers.Dense(64, activation='relu', name='b_blurb_title_dense')(b_combine_blurb_title)
    # b_combine_four_reshape shape = (?, 64)
    b_combine_four_reshape = keras.layers.Reshape([b_combine_four.shape[2]], name='b_combine_four_reshape')(b_combine_four)
    # 合并所有的书籍特征
    b_combine_book = keras.layers.concatenate([b_combine_blurb_title, b_combine_four_reshape], axis=1, name='b_combine_book')
    # 得到书籍矩阵
    b_feature_layer = keras.layers.Dense(200, name='b_feature_layer')(b_combine_book)
    return b_feature_layer


# 定义好我们所需要的所有features和targets
features = origin_DATA.features.values
targets=origin_DATA.labels.values

train_features, train_labels, val_features, val_lables, test_features, test_lables = get_train_val_test(features, targets)

"""
模型1
"""

class Net_works():
    def __init__(self, 
        batch_size=256, 
        epoch=5, 
        best_loss=999):
        self.batchsize = batch_size
        self.epoch = epoch
        self.best_loss = best_loss

    def creat_model(self):
        user_id, user_location, book_isbn, book_title, book_author, book_year, book_publisher, book_blurb = get_inputs()

        user_id_embedd, user_loca_embedd = get_user_embedding_16(user_id, user_location)
        
        book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd = get_book_embedding_16(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb)
        u_feature_layer = get_user_feature_lstm(user_id_embedd, user_loca_embedd)
        b_feature_layer = get_book_feature_lstm(book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd)
        multiply_layer = get_rating(u_feature_layer, b_feature_layer)
        model = keras.Model(inputs=[user_id, user_location, book_isbn, book_title, book_author, book_year, book_publisher, book_blurb], 
                    outputs=[multiply_layer])
        return model

    def train_model(self):
        model = self.creat_model()
        model_optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=model_optimizer, loss=keras.losses.mse)
        history = model.fit(train_features, train_labels, validation_data=(val_features, val_lables), epochs=self.epoch, batch_size=self.batchsize, verbose=1)
        return  history

    def predict_model(self, model):
        test_loss = self.model.evaluate(test_features, test_lables, verbose=0)
        return test_loss

if __name__ == "__main__":
    pass
else:
    pass