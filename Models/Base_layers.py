#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 17:24:28
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys, os

import re, pickle
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
import time
from tensorflow import keras

# 引入定义的超参hyperparameters
from Public import hyperparameters as hp
from Public import Config
# 引入数据预处理后的数据
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

def get_rating(user_feature, book_feature):
    multiply_layer = keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0]*layer[1],axis=1,keepdims=True), name = 'user_book_feature')((user_feature, book_feature))
    return multiply_layer

def get_inputs():
    # 用户特征输入
    user_id = keras.layers.Input(shape=(1, ), dtype='int32', name='user_id_input')
    user_location = keras.layers.Input(shape=(Config.LOCATION_LENTGH, ), dtype='int32', name='user_location_input')
    # 书籍特征输入
    book_isbn = keras.layers.Input(shape=(1, ), dtype='int32', name='book_isbn_input')
    book_author = keras.layers.Input(shape=(1, ), dtype='int32', name='book_author_input')
    book_year = keras.layers.Input(shape=(1, ), dtype='int32', name='book_year_input')
    book_publisher = keras.layers.Input(shape=(1, ), dtype='int32', name='book_publisher_input')  
    book_title = keras.layers.Input(shape=(Config.TITLE_LENGTH, ), dtype='int32', name='book_title_input')
    book_blurb = keras.layers.Input(shape=(Config.BLURB_LENGTH, ), dtype='int32', name='book_blurb_input')
    return user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb

# 构建User神经网络
def get_user_embedding(user_id, user_location):
    uid_embed_layer = tf.keras.layers.Embedding(all_user_id_number, hp.embedding_dim, input_length=1, name='uid_embed_layer')(user_id)
    location_embed_layer = tf.keras.layers.Embedding(all_location_words_number, hp.embedding_dim, input_length=location_length, name='location_embed_layer')(user_location)
    return uid_embed_layer, location_embed_layer

def get_user_embedding_half(u_id, u_loca):
    user_id_embedd = keras.layers.Embedding(all_user_id_number, hp.embedding_dim//2, name='user_id_embedding')(u_id)
    user_loca_embedd = keras.layers.Embedding(all_location_words_number, hp.embedding_dim//2 , name='user_loca_embedding')(u_loca)
    return user_id_embedd, user_loca_embedd

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
def get_book_embedding(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb):
    book_isbn_embed_layer = tf.keras.layers.Embedding(all_isbn_words_number, hp.embedding_dim, input_length = 1, name='book_isbn_embed_layer')(book_isbn)
    book_author_embed_layer = tf.keras.layers.Embedding(all_author_words_number, hp.embedding_dim, input_length=1, name='book_author_embed_layer')(book_author)
    book_year_embed_layer = tf.keras.layers.Embedding(all_year_words_number, hp.embedding_dim, input_length=1, name='book_year_embed_layer')(book_year)
    book_publisher_embed_layer = tf.keras.layers.Embedding(all_publisher_words_number, hp.embedding_dim, input_length = 1, name='book_publisher_embed_layer')(book_publisher)

    book_title_embed_layer = tf.keras.layers.Embedding(all_title_words_number, hp.embedding_dim, input_length=title_length, name='book_title_embed_layer')(book_title)
    book_blurb_embed_layer = tf.keras.layers.Embedding(all_blurb_words_number, hp.embedding_dim, input_length = blurb_length, name='book_blurb_embed_layer')(book_blurb)
    return book_isbn_embed_layer, book_author_embed_layer, book_year_embed_layer, book_publisher_embed_layer, book_title_embed_layer, book_blurb_embed_layer

def get_book_embedding_half(b_isbn, b_atuhor, b_year, b_publisher, b_title, b_blurb):
    book_isbn_embedd = keras.layers.Embedding(all_isbn_words_number, hp.embedding_dim//2, name='book_isbn_embedding')(b_isbn)
    book_author_embedd = keras.layers.Embedding(all_author_words_number, hp.embedding_dim//2, name='book_author_embedding')(b_atuhor)
    book_year_embedd = keras.layers.Embedding(all_year_words_number, hp.embedding_dim//2, name='book_year_embedding')(b_year)
    book_publisher_embedd = keras.layers.Embedding(all_publisher_words_number, hp.embedding_dim//2, name='book_publisher_embedding')(b_publisher)
    
    book_title_embedd = keras.layers.Embedding(all_title_words_number, hp.embedding_dim, name='book_title_embedding')(b_title)
    book_blurb_embedd = keras.layers.Embedding(all_blurb_words_number, hp.embedding_dim, name='book_blurb_embedding')(b_blurb)
    return book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd


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
    b_title_conv = keras.layers.Conv2D(filters=8, kernel_size=(2, hp.dense_dim), strides=1)(b_title_reshape)# shape=(?, 14, 1, 8)
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


