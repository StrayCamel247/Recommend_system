#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/03 19:17:30
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os

# 模型的环境
import re, pickle
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
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
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

blurb_embedding_dim = 50
def load_word2vec_martics(size=blurb_embedding_dim):
    # 这里min_count会对字典做截断，如果min_coun=5,那么出现次数小于5的词会被丢弃，需要更新原本的字典
    model=Word2Vec(sentences=origin_DATA.blurb2vect,size=size,min_count=0,window=5)
    # 得到向量矩阵
    blurb_word2vec_martics = np.zeros((len(origin_DATA.blurb2int), size))
    for i, value in origin_DATA.blurb2int.items():
        try:
            blurb_word2vec_martics[value, :] = model.wv[i]
        except:
            print(i ,value)
            print(model.wv[i])
            exit(1)
    print(blurb_word2vec_martics.shape)
    # 添加pad字符对应的向量
    blurb_word2vec_martics = np.insert(blurb_word2vec_martics, blurb_word2vec_martics.shape[0], axis=0, values=np.zeros(shape=(1, size)))
    print(blurb_word2vec_martics.shape)
    return blurb_word2vec_martics

blurb_word2vec_martics = load_word2vec_martics()

train_features, train_labels, val_features, val_lables, test_features, test_lables = get_train_val_test()
# 嵌入矩阵的维度
embed_dim = 4
embed_dim_title = 16

def user_embed_layer(u_id, u_loca):
    user_id_embedd = keras.layers.Embedding(all_user_id_number, embed_dim, name='user_id_embedding')(u_id)
    user_loca_embedd = keras.layers.Embedding(all_location_words_number, embed_dim , name='user_loca_embedding')(u_loca)
    return user_id_embedd, user_loca_embedd

def book_emded_layer(b_isbn, b_atuhor, b_year, b_publisher, b_title, b_blurb):
    book_isbn_embedd = keras.layers.Embedding(all_isbn_words_number, embed_dim, name='book_isbn_embedding')(b_isbn)
    book_author_embedd = keras.layers.Embedding(all_author_words_number, embed_dim, name='book_author_embedding')(b_atuhor)
    book_year_embedd = keras.layers.Embedding(all_year_words_number, embed_dim, name='book_year_embedding')(b_year)
    book_publisher_embedd = keras.layers.Embedding(all_publisher_words_number, embed_dim, name='book_publisher_embedding')(b_publisher)
    
    book_title_embedd = keras.layers.Embedding(all_title_words_number, embed_dim_title, name='book_title_embedding')(b_title)
    # 加载预训练模型
    book_blurb_embedd = keras.layers.Embedding(all_blurb_words_number, blurb_embedding_dim, name='book_blurb_embedding', weights=[blurb_word2vec_martics], trainable=True)(b_blurb)
    return book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd

def get_user_feature(u_id_embedd, u_loca_embedd):
    u_id_layer = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='u_id_dense')(u_id_embedd)
#     u_id_layer_drop = keras.layers.Dropout(rate=0.5, name='u_id_layer_drop')(u_id_layer)
    # u_id_layer.shape = (?, 1, 32)
    # u_loca_layer.shape = (?, 32)
    # 这里可以再加个Dense
    u_loca_layer = keras.layers.Bidirectional(tf.keras.layers.LSTM(16, name='u_loca_bilstm'), merge_mode='concat')(u_loca_embedd)
    u_loca_lstm_dense = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='u_loca_lstm_dense')(u_loca_layer)
    u_id_reshape = keras.layers.Reshape([32])(u_id_layer)
    u_combine = keras.layers.concatenate([u_id_reshape, u_loca_lstm_dense],axis=1, name='u_combine')
    print(u_combine.shape)
    # 这里能不能用激活函数
    u_feature_layer = keras.layers.Dense(100, activation='tanh', name='u_feature_layer')(u_combine)
    print(u_feature_layer.shape)
    return u_feature_layer

b_dense = 4
def get_book_feature(b_isbn_embedd, b_author_embedd, b_year_embedd, b_publisher_embedd, b_title_embedd, b_blurb_embedd):
    # 首先对前4个特征连接Dense层
    b_isbn_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='b_isbn_dense')(b_isbn_embedd)
    b_author_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='b_author_dense')(b_author_embedd)
    b_year_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='b_year_dense')(b_year_embedd)
    b_publisher_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='b_publisher_dense')(b_publisher_embedd)
    # 合并这四个特征,  b_combine_four shape = (?, 1, 16)
    b_combine_four = keras.layers.concatenate([b_isbn_dense, b_author_dense, b_year_dense, b_publisher_dense], name='b_four_combine')
    print('b_combine_four.shape', b_combine_four.shape)
    # 对title进行卷积
    b_title_reshape = keras.layers.Lambda(lambda layer: tf.expand_dims(layer, 3))(b_title_embedd)  # shape=(?,10, 16, 1)
    print('b_title_reshape.shape = ', b_title_reshape.shape)
    # b_title_conv.shape = 
    b_title_conv = keras.layers.Conv2D(filters=8, kernel_size=(2, embed_dim_title), kernel_regularizer=tf.nn.l2_loss, strides=1)(b_title_reshape)# shape=(?, 14, 1, 8)
    # b-title_pool.shape =
    b_title_pool = keras.layers.MaxPool2D(pool_size=(9, 1), strides=1)(b_title_conv) # shape=(?,1, 1, 8)
    print('b_title_conv.shape = ', b_title_conv)
    print('b_title_pool.shape = ', b_title_pool)
    
    # 对blurb进行处理
    # shape = 
    b_blurb_lstm_1 = keras.layers.Bidirectional(tf.keras.layers.LSTM(16, name='b_blurb_bilstm', dropout=0.5, return_sequences=True), merge_mode='concat')(b_blurb_embedd) 
    print('b_blurb_lstm_1.shape = ', b_blurb_lstm_1.shape)
    b_blurb_lstm_2 = keras.layers.LSTM(32, name='b_blurb_lstm', dropout=0.5, return_sequences=False)(b_blurb_lstm_1) 
    print('b_blurb_lstm_2.shape = ', b_blurb_lstm_2.shape)
    # 将title和blurb合并
    b_title_reshape = keras.layers.Reshape([b_title_pool.shape[3]])(b_title_pool)
    # b_combine_blurb_title.shape = 
    b_combine_blurb_title = keras.layers.concatenate([b_title_reshape, b_blurb_lstm_2], axis=1, name='b_combine_blurb_title')
    print('b_combine_blurb_title.shape = ', b_combine_blurb_title)
    b_blurb_title_dense = keras.layers.Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu', name='b_blurb_title_dense')(b_combine_blurb_title)
#     b_blurb_title_dense_drop = keras.layers.Dropout(rate=0.5, name='b_blurb_title_dense_drop')(b_blurb_title_dense)
    b_blurb_title_dense = keras.layers.Dense(64, activation='relu', name='b_blurb_title_dense')(b_combine_blurb_title)
    # b_combine_four_reshape shape = (?, 64)
    b_combine_four_reshape = keras.layers.Reshape([b_combine_four.shape[2]], name='b_combine_four_reshape')(b_combine_four)
    # 合并所有的书籍特征
#     b_combine_book = keras.layers.concatenate([b_combine_blurb_title, b_combine_four_reshape], axis=1, name='b_combine_book')
    b_combine_book = keras.layers.concatenate([b_blurb_title_dense, b_combine_four_reshape], axis=1, name='b_combine_book')

    # 得到书籍矩阵
    b_feature_layer = keras.layers.Dense(100, name='b_feature_layer', activation='tanh')(b_combine_book)
    return b_feature_layer

def get_rating(user_feature, book_feature):
#     multiply_layer = keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0]+layer[1], axis=1, keepdims=True), name = 'user_book_feature')((user_feature, book_feature))
    inference_layer = keras.layers.concatenate([user_feature, book_feature], axis=1, name='user_book_feature')
    inference_dense = tf.keras.layers.Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(inference_layer)
    multiply_layer = tf.keras.layers.Dense(1, name="inference")(inference_layer)  # inference_dense
    print(multiply_layer.shape)
    return multiply_layer

MODEL_DIR = './model/'

class model_network():
    def __init__(self):
        self.batchsize = 512
        self.epoch = 15

    def creat_model(self):
        user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb = get_inputs()
        user_id_embedd, user_loca_embedd = user_embed_layer(user_id, user_location)
        book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd = book_emded_layer(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb)
        u_feature_layer = get_user_feature(user_id_embedd, user_loca_embedd)
        b_feature_layer = get_book_feature(book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd)
        multiply_layer = get_rating(u_feature_layer, b_feature_layer)
        model = keras.Model(inputs=[user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb],
                    outputs=[multiply_layer])
        return model

    def train_model(self):
        weights_path = './model/model_2.hdf5'
#         checkpoint = keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor='val_loss', mode='min', save_weights_only=True)
        model_optimizer = keras.optimizers.Adamax()
        model = self.creat_model()
        model.compile(optimizer=model_optimizer, loss=keras.losses.mse)
        history = model.fit(train_features, train_labels, validation_data=(val_features, val_lables), epochs=self.epoch, batch_size=self.batchsize, verbose=1)
        return history

    def predict_model(self, model):
        test_loss = model.evaluate(test_features, test_lables, verbose=0)
        return test_loss









