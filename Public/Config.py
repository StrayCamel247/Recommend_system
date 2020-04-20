#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#__date__: 2020/04/02 15:45:53
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
import logging,datetime


# 打印时间的装饰器
def logging_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        print("this function <",func.__name__,">is running")
        res = func(*args, **kwargs)
        print("this function <",func.__name__,"> takes time：",datetime.datetime.now()-start)
        return res
    return wrapper

# 源数据的文件
ORIGIN_DATA_DIR = os.path.dirname(os.path.dirname(__file__))+'/Data/BX-CSV-Dump/'
# Data文件夹
DATA_DIR = os.path.dirname(os.path.dirname(__file__))+'/Data/'
# 缓存文件文件夹
FILTERED_DATA_DIR = os.path.dirname(os.path.dirname(__file__))+'/Tmp/'
# 模型存放文件
MODELS_DIR = os.path.dirname(os.path.dirname(__file__))+'/Models/'
# location title和blurb的取得长度值
LOCATION_LENTGH, TITLE_LENGTH, BLURB_LENGTH = 3, 15, 200

if __name__ == "__main__":
    print(os.path.dirname(os.path.dirname(__file__))+'/Tmp/')
