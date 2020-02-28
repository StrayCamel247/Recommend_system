#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os

import numpy as np 
import pandas as pd
import re

class dat_api():
    '''
    define the interface to data_files
    '''
    def __init__(self,
    bookmarks_path: 'the path to bookmarks.dat'=os.path.dirname(__file__)+'/data/bookmarks.dat'):
    #try set connection with dat files
        try:
            #acquire the 2000 rows data
            self.bookmarks = pd.read_csv(bookmarks_path, engine='python', nrows=2000, sep="\t", encoding='utf-8')
            # bookmarks'header is ['id','md5','title','url','md5Principal','urlPrincipal'],del row 'md5','md5Principal'
            self.bookmarks = self.bookmarks.drop(['id','md5','url','md5Principal'],axis=1)
            #将Title中的一些简单字符作预处理
            pattern = re.compile(r'[^a-zA-Z]')
            title_map = {val:re.sub(pattern, ' ', val) for ii,val in enumerate(set(self.bookmarks['title'])) }
            self.bookmarks['title'] = self.bookmarks['title'].map(title_map)
            genres_set = dict()
            
            for val in self.bookmarks['title'].str.split(' +'):
                for i in val:
                    if i not in genres_set:
                        genres_set[i]=1
                    else:
                        genres_set[i]+=1
            data = pd.DataFrame({'word': [val for val in genres_set.keys()],'counts': [val for val in genres_set.values()]})
            print(data)
            # 写入文件/save/bookmarks.dat  运行一次就好了
            # data.to_csv(os.path.dirname(__file__)+'/save/bookmarks.dat', sep=' ', header=True, index=True)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)

    
if __name__ == "__main__":
    test=dat_api()
