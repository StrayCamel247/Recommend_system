#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple

import sys,os
import numpy as np
import pandas as pd
import sqlite3
import re

class dat_api():
    '''
    define the interface to data_files
    '''
    def __init__(self,
    bookmarks_path: 'the path to bookmarks.dat'=os.path.dirname(__file__)+'/data/bookmarks.dat'):
    #try set connection with dat files
        try:
            #acquire the 1000 rows data
            self.bookmarks = pd.read_csv(bookmarks_path, engine='python', nrows=10, sep="\t")
            # bookmarks'header is ['id','md5','title','url','md5Principal','urlPrincipal'],del row 'md5','md5Principal'
            pattern = re.compile(r'^(.*)\((\d+)\)$')
            self.bookmarks = self.bookmarks.drop(['md5','md5Principal'],axis=1)
            # for booo in self.bookmarks:
            #     print(booo)
            print(self.bookmarks.iterrows())
            for k,v in self.bookmarks.iterrows():
                print(k,'----------------\n',v)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)

class sql_api():
    '''
    denfine the interface to sqlite3
    '''
    
    def __init__(self,
    db_path: 'the path to db'=os.path.dirname(__file__)+'/source/db.sqlite3'):
    #try set connection with db via db_path
        try:
            self.con = sqlite3.connect(db_path)
            self.cur = self.con.cursor()
            print("try set connection with db via db_path...")
            self.cur.execute("select * from web_novel")
            values = self.cur.fetchall()
            print(values)
        except sqlite3.Error as e:
            print("connect error|sqlite3 Error: %s" % e)
        
    def __del__(self):
        if self.con:
            self.con.close()
            print("connection disconnected...")
if __name__ == "__main__":
    # test_sql = sql_api()
    test_dat = dat_api()
