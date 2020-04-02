#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 22:16:56
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
    from .Base_layers import get_inputs, get_user_embedding_half, get_user_feature_lstm, get_book_embedding_half, get_book_feature_lstm
except ModuleNotFoundError:
    # 当本地运行测试的时候，需要使用直接from import，不然会报错ModuleNotFoundError: No module named '__main__.xxxxxx'; '__main__' is not a package
    from Base_layers import get_inputs, get_user_embedding_half, get_user_feature_lstm, get_book_embedding_half, get_book_feature_lstm, get_rating
    
# 引入数据预处理后的数据
from Data import origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number

# 定义好我们所需要的所有features和targets
m = len(origin_DATA.features['Location'])
# 对location取3位数
location = np.zeros((m, 3))
title = np.zeros((m, 15))
blurb = np.zeros((m, 200))
for i in range(m):
    location[i] = np.array(origin_DATA.features['Location'][i])
    title[i] = np.array(origin_DATA.features['Title'][i])
    blurb[i] = np.array(origin_DATA.features['Blurb'][i])

input_features = [origin_DATA.features['User-ID'].to_numpy(), location, 
                  origin_DATA.features['ISBN'].to_numpy(), origin_DATA.features['Author'].to_numpy(),
                 origin_DATA.features['Year'].to_numpy(), origin_DATA.features['Publisher'].to_numpy(), 
                 title, blurb]

labels = origin_DATA.labels.to_numpy()

"""
模型1
"""
MODEL_DIR = './model/'

class Net_works():
    def __init__(self, 
        batch_size=256,
        epoch=5, 
        best_loss=999):
        self.batchsize = batch_size
        self.epoch = epoch
        self.best_loss = best_loss

    def creat_model(self):
        user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb = get_inputs()
        user_id_embedd, user_loca_embedd = get_user_embedding_half(user_id, user_location)
        book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd = get_book_embedding_half(book_isbn, book_author, book_year, book_publisher, book_title, book_blurb)
        u_feature_layer = get_user_feature_lstm(user_id_embedd, user_loca_embedd)
        b_feature_layer = get_book_feature_lstm(book_isbn_embedd, book_author_embedd, book_year_embedd, book_publisher_embedd, book_title_embedd, book_blurb_embedd)
        multiply_layer = get_rating(u_feature_layer, b_feature_layer)
        model = keras.Model(inputs=[user_id, user_location, book_isbn, book_author, book_year, book_publisher, book_title, book_blurb],
                    outputs=[multiply_layer])
        return model

    def train_model(self):
        model = self.creat_model()
        model.compile(optimizer='adam', loss=keras.losses.mae)
        model.fit(input_features, labels, epochs=5, batch_size=512)
        print(model.summary())

if __name__ == "__main__":
    pass
else:
    pass