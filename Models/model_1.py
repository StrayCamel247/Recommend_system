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
    from .Base_layers import get_inputs, get_user_embedding_half, get_user_feature_lstm, get_book_embedding_half, get_book_feature_lstm, get_rating
except ModuleNotFoundError:
    # 当本地运行测试的时候，需要使用直接from import，不然会报错ModuleNotFoundError: No module named '__main__.xxxxxx'; '__main__' is not a package
    from Base_layers import get_inputs, get_user_embedding_half, get_user_feature_lstm, get_book_embedding_half, get_book_feature_lstm, get_rating
    
# 引入数据预处理后的数据
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

# 定义好我们所需要的所有features和targets
features = origin_DATA.features.values
targets=origin_DATA.labels.values

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

        user_id_embedd, user_loca_embedd = get_user_embedding_half(user_id, user_location)
        
        book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd = get_book_embedding_half(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb)
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