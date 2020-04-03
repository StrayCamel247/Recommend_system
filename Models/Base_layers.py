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
    book_title = keras.layers.Input(shape=(Config.TITLE_LENGTH, ), dtype='int32', name='book_title_input')
    book_author = keras.layers.Input(shape=(1, ), dtype='int32', name='book_author_input')
    book_year = keras.layers.Input(shape=(1, ), dtype='int32', name='book_year_input')
    book_publisher = keras.layers.Input(shape=(1, ), dtype='int32', name='book_publisher_input')
    book_blurb = keras.layers.Input(shape=(Config.BLURB_LENGTH, ), dtype='int32', name='book_blurb_input')
    return user_id, user_location, book_isbn, book_title, book_author, book_year, book_publisher, book_blurb


