#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 18:10:40
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Public import Config

import matplotlib.pyplot as plt
def history_show_loss(history):
    train_loss = history['loss']
    val_loss = history['val_loss']
    plt.figure(1)
    plt.plot(train_loss, c='r', label='train_loss')
    plt.plot(val_loss, c='b', label='val_loss')
    plt.legend()
    plt.xlim([0, 15])
    plt.savefig(Config.MODELS_DIR+"/model_0_history.png")
    plt.show()
# if __name__ == "__main__":
#     print(Config.FILTERED_DATA_DIR+'model_0_history.p')