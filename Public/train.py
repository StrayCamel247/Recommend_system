#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 18:07:25
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
import re, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Public import Config

def model_history(model, version):
    path = Config.FILTERED_DATA_DIR+"{}_model_history.p".format(version)
    print(path)
    try:
        # 从缓存的文件夹FILTERED_DATA_DIR获取基本被过滤后的文件
        history= pickle.load(open(path, mode="rb+"))
        return history
    except:
        # 如果缓存的文件不存在或者没有，则在源目录ORIGIN_DATA_DIR获取
        history = model.train_model()
        history = history.history
        pickle.dump((history), open(path, "wb"))
        return history
