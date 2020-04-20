#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 16:55:22
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Public import Config, model_history, data_analysis
from Public import data_analysis as mat_show
import Models

def run(net_model):
    model = net_model.Net_works(256,10)
    history = model_history(model, 0)
    mat_show.history_show_loss(history, 0)

# 开始训练前请先解压Data中的BX-CSV-Dump数据集
# 第一次训练会缓存训练结果到tmp文件夹，如果更改数据重新训练请删除对应的history文件，再运行

run(Models.model_0)
# main(model_1, 1)
# model_2 会报错 修复ing
# main(model_2, 2)