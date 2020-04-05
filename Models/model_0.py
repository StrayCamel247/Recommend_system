#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 15:45:40
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
    
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

# 构建User神经网络
def get_user_embedding(user_id, user_location):
    uid_embed_layer = tf.keras.layers.Embedding(all_user_id_number, hp.embedding_dim, input_length=1, name='uid_embed_layer')(user_id)
    location_embed_layer = tf.keras.layers.Embedding(all_location_words_number, hp.embedding_dim, input_length=location_length, name='location_embed_layer')(user_location)
    return uid_embed_layer, location_embed_layer

def get_user_feature_gru(uid_embed_layer, location_embed_layer):
    """
    model_0: 使用gru处理用户信息中的location特征
    """
    #第一层全连接
    uid_fc_layer = tf.keras.layers.Dense(hp.dense_dim, name='uid_fc_layer', activation='relu')(uid_embed_layer)
    location_fc_layer = tf.keras.layers.Dense(hp.dense_dim, name='location_fc_layer', activation='relu')(location_embed_layer)
    #对location进行Encoder提取特征
    location_gru_layer = tf.keras.layers.GRU(units=hp.dense_dim, dropout=hp.dropout_keep, name='location_gru_layer')(location_fc_layer)
    #[None, 32]
    print(location_gru_layer.shape)
    location_gru_expand_layer = tf.expand_dims(location_gru_layer, axis=1)
    
    #第二层全连接
    user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer, location_gru_expand_layer], 2)
    user_dense_layer = tf.keras.layers.Dense(Config.BLURB_LENGTH, activation='tanh', name='user_dense_layer')(user_combine_layer)
    user_dense_layer_flat = tf.keras.layers.Reshape([Config.BLURB_LENGTH], name="user_combine_layer_flat")(user_dense_layer)
    return user_dense_layer, user_dense_layer_flat

# 构建Book神经网络
def get_book_embedding(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb):
    book_isbn_embed_layer = tf.keras.layers.Embedding(all_isbn_words_number, hp.embedding_dim, input_length = 1, name='book_isbn_embed_layer')(book_isbn)
    book_author_embed_layer = tf.keras.layers.Embedding(all_author_words_number, hp.embedding_dim, input_length=1, name='book_author_embed_layer')(book_author)
    book_year_embed_layer = tf.keras.layers.Embedding(all_year_words_number, hp.embedding_dim, input_length=1, name='book_year_embed_layer')(book_year)
    book_publisher_embed_layer = tf.keras.layers.Embedding(all_publisher_words_number, hp.embedding_dim, input_length = 1, name='book_publisher_embed_layer')(book_publisher)

    book_title_embed_layer = tf.keras.layers.Embedding(all_title_words_number, hp.embedding_dim, input_length=title_length, name='book_title_embed_layer')(book_title)
    book_blurb_embed_layer = tf.keras.layers.Embedding(all_blurb_words_number, hp.embedding_dim, input_length = blurb_length, name='book_blurb_embed_layer')(book_blurb)
    return book_isbn_embed_layer, book_author_embed_layer, book_year_embed_layer, book_publisher_embed_layer, book_title_embed_layer, book_blurb_embed_layer

def get_book_feature_gru(book_isbn_embed_layer, book_author_embed_layer, book_year_embed_layer, book_publisher_embed_layer, book_title_embed_layer, book_blurb_embed_layer):
    """
    model_0: 使用gru处理书籍信息中的blurb特征
    """
    #  对isbn, author, year, publisher第一层全连接
    book_isbn_dense_layer = tf.keras.layers.Dense(hp.dense_dim, activation='relu', name='book_isbn_dense_layer')(book_isbn_embed_layer)
    book_author_dense_layer = tf.keras.layers.Dense(hp.dense_dim, activation='relu', name='book_author_dense_layer')(book_author_embed_layer)
    book_year_dense_layer = tf.keras.layers.Dense(hp.dense_dim, activation='relu', name='book_year_dense_layer')(book_year_embed_layer)
    book_publisher_dense_layer = tf.keras.layers.Dense(hp.dense_dim, activation='relu', name='book_publisher_dense_layer')(book_publisher_embed_layer)
    book_title_embed_layer_expand = tf.expand_dims(book_title_embed_layer, axis=-1)
    #  对title进行文本卷积
    #  book_title_embed_layer_expand:[None, 15, 16, 1]
    #  对文本嵌入层使用不同的卷积核做卷积核最大池化
    pool_layer_list = []
    for window_size in hp.window_sizes:
        title_conv_layer = tf.keras.layers.Conv2D(filters = hp.filter_num, kernel_size = (window_size, hp.embedding_dim), strides=1, activation='relu')(book_title_embed_layer_expand)
        title_maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(title_length-window_size+1, 1), strides=1)(title_conv_layer)
        pool_layer_list.append(title_maxpool_layer)
    pool_layer_layer = tf.keras.layers.concatenate(pool_layer_list, axis=-1, name='title_pool_layer')
    max_num = len(hp.window_sizes)*hp.filter_num
    pool_layer_flat = tf.keras.layers.Reshape([1, max_num], name='pool_layer_flat')(pool_layer_layer)
    dropout_layer = tf.keras.layers.Dropout(hp.dropout_keep, name = "dropout_layer")(pool_layer_flat)

    # 对简介进行Encoder特征提取
    book_blurb_dense_layer = tf.keras.layers.Dense(hp.dense_dim, activation='relu', name='book_blurb_dense_layer')(book_blurb_embed_layer)
    book_blurb_gru_layer = tf.keras.layers.GRU(units=hp.dense_dim, dropout=hp.dropout_keep, name='book_blurb_gru_layer')(book_blurb_dense_layer)
    print('book_blurb_gru_layer=', book_blurb_gru_layer.shape)
    book_blurb_gru_expand_layer = tf.expand_dims(book_blurb_gru_layer, axis=1)
    book_combine_layer = tf.keras.layers.concatenate([book_isbn_dense_layer, book_author_dense_layer, book_year_dense_layer, book_publisher_dense_layer, dropout_layer, book_blurb_gru_expand_layer], axis=-1)
    book_dense_layer = tf.keras.layers.Dense(Config.BLURB_LENGTH, activation='tanh')(book_combine_layer)
    book_dense_layer_flat = tf.keras.layers.Reshape([Config.BLURB_LENGTH], name="book_dense_layer_flat")(book_dense_layer)
    return book_dense_layer, book_dense_layer_flat

# 定义好我们所需要的所有features和targets
features = origin_DATA.features.values
targets = origin_DATA.labels.values

train_features, train_labels, val_features, val_lables, test_features, test_lables = get_train_val_test(features, targets)

class Net_works(object):
    def __init__(self, 
        batch_size=256, 
        epoch=5):
        self.batchsize = batch_size
        self.epoch = epoch
         # 获取输入占位符
        user_id, user_location, book_isbn, book_title, book_author, book_year, book_publisher, book_blurb = get_inputs()
        # 获取User的2个嵌入向量
        uid_embed_layer, location_embed_layer = get_user_embedding(user_id, user_location)
        # 得到用户特征
        user_dense_layer, user_dense_layer_flat =get_user_feature_gru(uid_embed_layer, location_embed_layer)
        # 获取书籍的嵌入向量
        book_isbn_embed_layer, book_author_embed_layer, book_year_embed_layer, book_publisher_embed_layer, book_title_embed_layer, book_blurb_embed_layer=get_book_embedding(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb)
        # 获取书籍特征
        book_dense_layer, book_dense_layer_flat=get_book_feature_gru(book_isbn_embed_layer, book_author_embed_layer, book_year_embed_layer, book_publisher_embed_layer, book_title_embed_layer, book_blurb_embed_layer)
        
        # 计算出评分
        # 将用户特征和电影特征做矩阵乘法得到一个预测评分的方案
        print("user_dense_layer_flat=", user_dense_layer_flat.shape)
        print("book_dense_layer_flat=", book_dense_layer_flat.shape)
        inference = get_rating(user_dense_layer_flat, book_dense_layer_flat)

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