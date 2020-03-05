#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel,EvaGalois
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os

import threading
# 线程池
from concurrent.futures import ThreadPoolExecutor
import numpy as np 
import pandas as pd
import re

class DataLoad:
    '''
    tags 默认 ['id', 'value']
    bookmarks 默认 ['id', 'md5', 'title', 'url' ,'md5Principal', 'urlPrincipal']
    bookmark_tags 默认 ['bookmarkID', 'tagID', 'tagWeight']
    user_contacts 默认 ['userID', 'contactID', 'date_day', 'date_month', 'date_year', 'date_hour', 'date_minute', 'date_second']
    user_taggedbookmarks 默认 ['userID', 'bookmarkID', 'tagID', 'day', 'month', 'year', 'hour', 'minute', 'second']
    '''
    def __init__(self, dataName):
        self.datafile = os.path.dirname(os.path.abspath(__file__)) + '/data/' + dataName + '.dat'
        self.table_title = pd.read_table(self.datafile, sep='\t', header=None, nrows=1, engine='python')
        self.table = pd.read_table(self.datafile, sep='\t', header=None, skiprows=[0], names=np.array(self.table_title)[0], engine='python')



class PreData_api():
    '''
    define the interface to data_files, then do some preprocessing on source data.
    '''
    def __init__(self,
    new_paths: 'keys: the origin_files paths; values: the new_files paths' = [
        '/save/bags_of_title.csv',
        '/save/books.csv',
        '/save/tags.csv',
        '/save/users.csv',
    ],
    bookmarks_path: 'the path to bookmarks.dat' = os.path.dirname(__file__)+'/data/bookmarks.dat'):
        # 若预处理后的文件存在,则无需进行预处理
        for _ in new_paths:
            path = os.path.dirname(__file__)+_
            if not os.path.exists(path):
                print('Data is not preprocessed completely...')
                self.generate_preprocessed_files(new_paths)
                break

        #try set connection with dat files
        try:
            #acquire the 2000 rows data,bookmarks'header is ['id','md5','title','url','md5Principal','urlPrincipal']
            self.bookmarks = pd.read_csv(bookmarks_path, engine='python', nrows=2000, sep="\t", encoding='utf-8').drop(['md5','url','md5Principal'],axis=1)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)

    def generate_preprocessed_files(self,tmp_path):
        data=[]
        try:
            #acquire the 2000 rows data,bookmarks'header is ['id','md5','title','url','md5Principal','urlPrincipal']
            self.bookmarks = pd.read_csv(os.path.dirname(__file__)+'/data/bookmarks.dat', engine='python', nrows=2000, sep="\t", encoding='utf-8').drop(['md5','url','md5Principal'],axis=1)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)
        
        
        
        #将Title中的一些简单字符作预处理，对书本名称作词袋处理
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
        data.append(pd.DataFrame({'id': [i for i in range(len(genres_set)) ],'word': [val for val in genres_set.keys()],'counts': [val for val in genres_set.values()]}))
        
        # 对书籍和书籍标签、用户，信息进行整合和预处理
        tags = DataLoad('tags').table
        bookmarks_data = DataLoad('bookmarks').table
        bookmarks = bookmarks_data.drop(['md5', 'url', 'md5Principal', 'urlPrincipal'], axis=1)
        bookmark_tags = DataLoad('bookmark_tags').table
        user_contacts_data = DataLoad('user_contacts').table
        user_contacts = user_contacts_data.drop(['date_day', 'date_month', 'date_year', 'date_hour', 'date_minute', 'date_second'], axis=1)
        user_taggedbookmarks_data = DataLoad('user_taggedbookmarks').table
        user_taggedbookmarks = user_taggedbookmarks_data.drop(['day', 'month', 'year', 'hour', 'minute', 'second'], axis=1)


        for i in range(0, len(user_contacts['contactID'])):
            if user_contacts['contactID'][i] >= 1000:
                user_contacts['contactID'][i] = round(user_contacts['contactID'][i] / 100)


        book = pd.merge(bookmarks, bookmark_tags, left_on='id', right_on='bookmarkID', how='left')
        Books = book.drop(['bookmarkID'], axis=1)
        Books.replace(np.nan, 0, inplace=True)
        Books.replace(np.inf, 0, inplace=True)
        Books['tagID_int']=Books['tagID'].astype(int)
        Books['tagWeight_int']=Books['tagWeight'].astype(int)
        Books = Books.drop(['tagID', 'tagWeight'], axis=1)
        Books = Books.rename(columns={'tagID_int':'tagID', 'tagWeight_int':'tagWeight'})
        data.append(Books)
        # Books.to_csv(os.path.dirname(__file__)+'/save/book.csv', sep=' ', header=True, index=True)
        # Books.to_csv(os.path.dirname(__file__)+'/save/book.zip', index=False, compression=dict(method='zip', archive_name='book.csv'))

        Tags = tags.astype({'value':str})
        # Tags.to_csv(os.path.dirname(__file__)+'/save/tag.csv', sep=' ', header=True, index=True)
        # Tags.to_csv(os.path.dirname(__file__)+'/save/tag.zip', index=False, compression=dict(method='zip', archive_name='tag.csv'))
        data.append(Tags)
        user = pd.merge(user_contacts, user_taggedbookmarks, on='userID', how='left')
        Users = user.rename(columns={'bookmarkID': 'BookID'})
        # Users.to_csv(os.path.dirname(__file__)+'/save/user.csv', sep=' ', header=True, index=True)
        # Users.to_csv(os.path.dirname(__file__)+'/save/user.zip', index=False, compression=dict(method='zip', archive_name='user.csv'))
        data.append(Users)
        


        def saving(processed_data, path):
            processed_data.to_csv(path, sep=' ', header=True, index=False)
        executor = ThreadPoolExecutor(max_workers=20)
        for d,p in zip(data,tmp_path):
            # 利用线程池--concurrent.futures模块来管理多线程：
            future = executor.submit(saving,d,os.path.dirname(__file__)+p)
            print(d,p)
        print("Data is preprocessed completely!!!")
    
if __name__ == "__main__":
    for a,b in zip([1,2],[7,8]):
        print(a,b)