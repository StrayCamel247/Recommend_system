#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 17:24:28
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os

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

def get_inputs():
    # 用户特征输入
    user_id = keras.layers.Input(shape=(1,), dtype='int32', name='user_id_input')
    user_location = keras.layers.Input(shape=(Config.LOCATION_LENTGH,), dtype='int32', name='user_location_input')
    # 书籍特征输入
    book_isbn = keras.layers.Input(shape=(1,),  dtype='int32', name='book_isbn_input')
    book_author = keras.layers.Input(shape=(1,),  dtype='int32', name='book_author_input')
    book_year = keras.layers.Input(shape=(1,),  dtype='int32', name='book_year_input')
    book_publisher = keras.layers.Input(shape=(1,),  dtype='int32', name='book_publisher_input')  
    book_title = keras.layers.Input(shape=(Config.TITLE_LENGTH, ), dtype='int32', name='book_title_input')
    book_blurb = keras.layers.Input(shape=(Config.BLURB_LENGTH, ), dtype='int32', name='book_blurb_input')
    return user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb

"""
构建User神经网络
"""
def get_user_embedding(user_id,user_location):
    uid_embed_layer = tf.keras.layers.Embedding(all_user_id_number,hp.embedding_dim,input_length=1,name='uid_embed_layer')(user_id)
    location_embed_layer = tf.keras.layers.Embedding(all_location_words_number,hp.embedding_dim,input_length=location_length,name='location_embed_layer')(user_location)
    return uid_embed_layer, location_embed_layer

def get_user_feature_layer(uid_embed_layer,location_embed_layer):
#     第一层全连接
    uid_fc_layer = tf.keras.layers.Dense(hp.dense_dim,name='uid_fc_layer',activation='relu')(uid_embed_layer)
    location_fc_layer = tf.keras.layers.Dense(hp.dense_dim,name='location_fc_layer',activation='relu')(location_embed_layer)
#  对location进行Encoder提取特征
    location_gru_layer = tf.keras.layers.GRU(units=hp.dense_dim,dropout=hp.dropout_keep,name='location_gru_layer')(location_fc_layer)
#     [None,32]
    print(location_gru_layer.shape)
    location_gru_expand_layer = tf.expand_dims(location_gru_layer,axis=1)
    
#     第二层全连接
    user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer,location_gru_expand_layer],2)
    user_dense_layer = tf.keras.layers.Dense(Config.BLURB_LENGTH,activation='tanh',name='user_dense_layer')(user_combine_layer)
    user_dense_layer_flat = tf.keras.layers.Reshape([Config.BLURB_LENGTH], name="user_combine_layer_flat")(user_dense_layer)
    return user_dense_layer,user_dense_layer_flat

"""
构建book神经网络
"""
def get_book_embedding(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb):
    book_isbn_embed_layer = tf.keras.layers.Embedding(all_isbn_words_number,hp.embedding_dim,input_length = 1,name='book_isbn_embed_layer')(book_isbn)
    book_author_embed_layer = tf.keras.layers.Embedding(all_author_words_number,hp.embedding_dim,input_length=1,name='book_author_embed_layer')(book_author)
    book_year_embed_layer = tf.keras.layers.Embedding(all_year_words_number,hp.embedding_dim,input_length=1,name='book_year_embed_layer')(book_year)
    book_publisher_embed_layer = tf.keras.layers.Embedding(all_publisher_words_number,hp.embedding_dim,input_length = 1,name='book_publisher_embed_layer')(book_publisher)
    book_title_embed_layer = tf.keras.layers.Embedding(all_title_words_number,hp.embedding_dim,input_length=title_length,name='book_title_embed_layer')(book_title)
    book_blurb_embed_layer = tf.keras.layers.Embedding(all_blurb_words_number,hp.embedding_dim,input_length = blurb_length,name='book_blurb_embed_layer')(book_blurb)
    return book_isbn_embed_layer,book_author_embed_layer,book_year_embed_layer,book_publisher_embed_layer,book_title_embed_layer,book_blurb_embed_layer
    
def get_book_feature_layer(book_isbn_embed_layer,book_author_embed_layer,book_year_embed_layer,book_publisher_embed_layer,book_title_embed_layer,book_blurb_embed_layer):
#     对isbn,author,year,publisher第一层全连接
    book_isbn_dense_layer = tf.keras.layers.Dense(hp.dense_dim,activation='relu',name='book_isbn_dense_layer')(book_isbn_embed_layer)
    book_author_dense_layer = tf.keras.layers.Dense(hp.dense_dim,activation='relu',name='book_author_dense_layer')(book_author_embed_layer)
    book_year_dense_layer = tf.keras.layers.Dense(hp.dense_dim,activation='relu',name='book_year_dense_layer')(book_year_embed_layer)
    book_publisher_dense_layer = tf.keras.layers.Dense(hp.dense_dim,activation='relu',name='book_publisher_dense_layer')(book_publisher_embed_layer)
    book_title_embed_layer_expand = tf.expand_dims(book_title_embed_layer,axis=-1)
#     对title进行文本卷积
#     book_title_embed_layer_expand:[None,15,16,1]
#     对文本嵌入层使用不同的卷积核做卷积核最大池化
    pool_layer_list = []
    for window_size in hp.window_sizes:
        title_conv_layer = tf.keras.layers.Conv2D(filters = hp.filter_num,kernel_size = (window_size,hp.embedding_dim),strides=1,activation='relu')(book_title_embed_layer_expand)
        title_maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(title_length-window_size+1,1),strides=1)(title_conv_layer)
        pool_layer_list.append(title_maxpool_layer)
    pool_layer_layer = tf.keras.layers.concatenate(pool_layer_list,axis=-1,name='title_pool_layer')
    max_num = len(hp.window_sizes)*hp.filter_num
    pool_layer_flat = tf.keras.layers.Reshape([1,max_num],name='pool_layer_flat')(pool_layer_layer)
    dropout_layer = tf.keras.layers.Dropout(hp.dropout_keep, name = "dropout_layer")(pool_layer_flat)

    # 对简介进行Encoder特征提取
    book_blurb_dense_layer = tf.keras.layers.Dense(hp.dense_dim,activation='relu',name='book_blurb_dense_layer')(book_blurb_embed_layer)
    book_blurb_gru_layer = tf.keras.layers.GRU(units=hp.dense_dim,dropout=hp.dropout_keep,name='book_blurb_gru_layer')(book_blurb_dense_layer)
    print('book_blurb_gru_layer=',book_blurb_gru_layer.shape)
    book_blurb_gru_expand_layer = tf.expand_dims(book_blurb_gru_layer,axis=1)
    book_combine_layer = tf.keras.layers.concatenate([book_isbn_dense_layer,book_author_dense_layer,book_year_dense_layer,book_publisher_dense_layer,dropout_layer,book_blurb_gru_expand_layer],axis=-1)
    book_dense_layer = tf.keras.layers.Dense(Config.BLURB_LENGTH, activation='tanh')(book_combine_layer)
    book_dense_layer_flat = tf.keras.layers.Reshape([Config.BLURB_LENGTH], name="book_dense_layer_flat")(book_dense_layer)
    return book_dense_layer,book_dense_layer_flat