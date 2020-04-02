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
from Models import model_0, model_1

def main(net_model, m_version):
    model = net_model.Net_works(256,10)
    history = model_history(model, m_version)
    mat_show.history_show_loss(history, m_version)
# 第一次训练会缓存训练结果到tmp文件夹，如果更改数据重新训练请删除对应的history文件，再运行
# main(model_0, 0)
main(model_1, 1)
