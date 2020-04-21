#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ : stray_camel
# __date__: 2020/04/03 19:17:30
# pip_source : https://mirrors.aliyun.com/pypi/simple
import sys, os
import math
from tqdm import tqdm
# 模型的环境
import re, pickle
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
import time
from tensorflow import keras
from transformers import BertTokenizer, BertConfig, TFBertModel


# for module in tf, np, pd, sklearn, tf, keras:
#     print(module.__name__, module.__version__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入定义的超参hyperparameters
from Public import hyperparameters as hp
from Public import Config

# 引入我们构建神经网络所需的层

try:
    # 当外部调用同目录包的时候需要使用相对路径访问
    from .Base_layers import get_rating
except ModuleNotFoundError:
    # 当本地运行测试的时候，需要使用直接from import，不然会报错ModuleNotFoundError: No module named '__main__.xxxxxx'; '__main__' is not a package
    from Base_layers import get_rating

# 引入数据预处理后的数据
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, \
    all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, \
    blurb_length, all_blurb_words_number



max_sequence_length = 512
bert_path = './bert_models/'
# title+blurb的最大长度
blurb_series = origin_DATA.features.Blurb
title_series = origin_DATA.features.Title

#  预训练模型来自于 https://github.com/huggingface/transformers
class pre_title_blurb():
    def __init__(self, lengths):
        self.max_sequence_length = lengths
        self.bert_path = bert_path
        self.blurb_series = origin_DATA.features.Blurb
        self.title_series = origin_DATA.features.Title
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path+'bert-base-uncased-vocab.txt')
        self.input_ids, self.input_type_ids, self.input_masks = self.return_id()
    def return_id(self):
        input_ids, input_type_ids, input_masks = [], [], []
        for index in tqdm(range(math.ceil(blurb_series.shape[0]/10))):
            # 这个encode_plus可以帮助我们自动标记text,pos,segement的token
            inputs = self.tokenizer.encode_plus(text=title_series[index], text_pair=blurb_series[index], add_special_tokens=True,
                                           max_length=self.max_sequence_length, truncation_strategy='longest_first')
            # 我们需要手动补齐,得到三个向量对应的token
            padding_id = self.tokenizer.pad_token_id
            input_id = inputs['input_ids']
            padding_length = self.max_sequence_length-len(input_id)
            input_id = inputs['input_ids'] + [padding_id] * (padding_length)
            input_type_id = inputs['token_type_ids']
            input_type_id = input_type_id + [0] * padding_length
            input_mask = inputs['attention_mask']
            input_mask = input_mask + [0] * padding_length
            input_ids.append(input_id)
            input_type_ids.append(input_type_id)
            input_masks.append(input_mask)
        return np.array(input_ids), np.array(input_type_ids), np.array(input_masks)

data_blurb_title = pre_title_blurb(lengths=max_sequence_length)

print(data_blurb_title.input_ids.shape)
print(data_blurb_title.input_masks.shape)
print(data_blurb_title.input_type_ids.shape)


def get_inputs_bert():
    # 用户特征输入
    user_id = keras.layers.Input(shape=(1,), dtype='int32', name='user_id_input')
    user_location = keras.layers.Input(shape=(3,), dtype='int32', name='user_location_input')

    book_isbn = keras.layers.Input(shape=(1,), dtype='int32', name='book_isbn_input')
    book_author = keras.layers.Input(shape=(1,), dtype='int32', name='book_author_input')
    book_year = keras.layers.Input(shape=(1,), dtype='int32', name='book_year_input')
    book_publisher = keras.layers.Input(shape=(1,), dtype='int32', name='book_publisher_input')
    book_title_blurb_id = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name='book_title_blurb_id')
    book_title_blurb_type_id = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name='book_title_blurb_type_id')
    book_title_blurb_mask = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name='book_title_blurb_mask')
    return user_id, user_location,  book_isbn, book_author, book_year, book_publisher, book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask

# 嵌入矩阵的维度
embed_dim = 8
embed_dim_words = 16

def user_embed_layer(u_id, u_loca):
    user_id_embedd = keras.layers.Embedding(all_user_id_number, embed_dim, name='user_id_embedding')(u_id)
    user_loca_embedd = keras.layers.Embedding(all_location_words_number, embed_dim , name='user_loca_embedding')(u_loca)
    return user_id_embedd, user_loca_embedd


def book_emded_layer(b_isbn, b_atuhor, b_year, b_publisher):
    book_isbn_embedd = keras.layers.Embedding(all_isbn_words_number, embed_dim, name='book_isbn_embedding')(b_isbn)
    book_author_embedd = keras.layers.Embedding(all_author_words_number, embed_dim, name='book_author_embedding')(b_atuhor)
    book_year_embedd = keras.layers.Embedding(all_year_words_number, embed_dim, name='book_year_embedding')(b_year)
    book_publisher_embedd = keras.layers.Embedding(all_publisher_words_number, embed_dim, name='book_publisher_embedding')(
        b_publisher)
    return book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd

def get_user_feature(u_id_embedd, u_loca_embedd):
    u_id_layer = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='u_id_dense')(u_id_embedd)
    # 这里可以再加个Dense
    u_loca_layer = keras.layers.LSTM(16, go_backwards=True, name='u_loca_lstm')(u_loca_embedd)
    u_loca_layer_lstm = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='u_loca_layer_lstm')(u_loca_layer)
    u_id_reshape = keras.layers.Reshape([32])(u_id_layer)
    u_combine = keras.layers.concatenate([u_id_reshape, u_loca_layer_lstm],axis=1, name='u_combine')
    # 这里能不能用激活函数
    u_feature_layer = keras.layers.Dense(200, activation='tanh', name='u_feature_layer')(u_combine)
    return u_feature_layer

b_dense = 4
def get_book_feature(b_isbn_embedd, b_author_embedd, b_year_embedd, b_publisher_embedd, book_title_id, book_title_type_id, book_title_mask):
    # 首先对前4个特征连接Dense层
    b_isbn_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss,
                                      name='b_isbn_dense')(b_isbn_embedd)
    b_author_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss,
                                        name='b_author_dense')(b_author_embedd)
    b_year_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss,
                                      name='b_year_dense')(b_year_embedd)
    b_publisher_dense = keras.layers.Dense(b_dense, activation='relu', kernel_regularizer=tf.nn.l2_loss,
                                           name='b_publisher_dense')(b_publisher_embedd)
    # 合并这四个特征,  b_combine_four shape = (?, 1, 16)
    b_combine_four = keras.layers.concatenate([b_isbn_dense, b_author_dense, b_year_dense, b_publisher_dense],
                                              name='b_four_combine')
    print('b_combine_four.shape', b_combine_four.shape)
    b_combine_four_reshape = keras.layers.Reshape([b_combine_four.shape[2]], name='b_combine_four_reshape')(b_combine_four)

    config = BertConfig()
    # 获取隐藏层的信息
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(bert_path + 'bert-base-uncased-tf_model.h5', config=config)
    book_title_cls = bert_model(book_title_id, attention_mask=book_title_mask, token_type_ids=book_title_type_id)
    print(len(book_title_cls))
    print(book_title_cls[0].shape)
    print(book_title_cls[1].shape)
    book_feature_layer = keras.layers.Dense(64, activation='tanh')(book_title_cls[1])
    b_combine_book = keras.layers.concatenate([book_feature_layer, b_combine_four_reshape], axis=1, name='b_combine_book')
    # 得到书籍矩阵
    b_feature_layer = keras.layers.Dense(200, name='b_feature_layer', activation='tanh')(b_combine_book)
    return b_feature_layer


def get_train_val_test_bert():
    m = len(origin_DATA.features['Location'])
    m = math.ceil(m/10)
    # 对location取3位数
    loca = np.zeros((m, 3))
    for i in range(m):
        loca[i] = np.array(origin_DATA.features['Location'][i])

    print(loca[:-2])
    input_features = [origin_DATA.features['User-ID'].to_numpy(), loca,
                      origin_DATA.features['ISBN'].to_numpy(), origin_DATA.features['Author'].to_numpy(),
                      origin_DATA.features['Year'].to_numpy(), origin_DATA.features['Publisher'].to_numpy(),
                     data_blurb_title.input_ids, data_blurb_title.input_masks, data_blurb_title.input_type_ids]
    labels = origin_DATA.labels.to_numpy()
    # 分割数据集以及shuffle
    np.random.seed(100)
    number_features = len(input_features)
    shuffle_index = np.random.permutation(m)
    shuffle_train_index = shuffle_index[:math.ceil(m*0.96)]
    shuffle_val_index = shuffle_index[math.ceil(m*0.96): math.ceil(m*0.98)]
    shuffle_test_index = shuffle_index[math.ceil(m*0.98):]
    train_features = [input_features[i][shuffle_train_index] for i in range(number_features)]
    train_labels = labels[shuffle_train_index]
    val_features = [input_features[i][shuffle_val_index] for i in range(number_features)]
    val_lables = labels[shuffle_val_index]
    test_features = [input_features[i][shuffle_test_index] for i in range(number_features)]
    test_lables = labels[shuffle_test_index]
    return train_features, train_labels, val_features, val_lables, test_features, test_lables


train_features, train_labels, val_features, val_lables, test_features, test_lables = get_train_val_test_bert()
print(train_features[0].shape)
print(val_features[0].shape)
print(test_features[0].shape)


class Net_works():
    def __init__(self, batch_size, epochs):
        self.batchsize = batch_size
        self.epoch = epochs
    def creat_model(self):
        user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title_blurb_id, \
        book_title_blurb_type_id, book_title_blurb_mask = get_inputs_bert()
        user_id_embedd, user_loca_embedd = user_embed_layer(user_id, user_location)
        book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd = \
            book_emded_layer(book_isbn, book_author, book_year, book_publisher)

        u_feature_layer = get_user_feature(user_id_embedd, user_loca_embedd)
        b_feature_layer = get_book_feature(book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd,
                                           book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask)
        multiply_layer = get_rating(u_feature_layer, b_feature_layer)
        model = keras.Model(inputs=[user_id, user_location,  book_isbn, book_author, book_year, book_publisher,
                                    book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask],
                            outputs=[multiply_layer])
        return model
    def train_model(self):
        model_optimizer = keras.optimizers.Adam()
        model = self.creat_model()
        model.compile(optimizer=model_optimizer, loss=keras.losses.mse)
        history = model.fit(train_features, train_labels, validation_data=(val_features, val_lables), epochs=self.epoch, batch_size=self.batchsize, verbose=1)
        print(model.summary())
        keras.utils.plot_model(model, to_file='./Models/Structure/model_3.png', show_shapes=True, show_layer_names=True)
        return history
    def predict_model(self, model):
        test_loss = model.evaluate(test_features, test_lables, batch_size=self.batchsize, verbose=1)
        return test_loss