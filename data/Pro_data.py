#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple

import os, sys, re, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Public import Config
import numpy as np
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords
# 线程池
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED


# 源数据的文件
ORIGIN_DATA_DIR = os.path.dirname(os.path.dirname(__file__))+'/Data/BX-CSV-Dump/'
# 缓存文件文件夹
FILTERED_DATA_DIR = os.path.dirname(os.path.dirname(__file__))+'/Tmp/'
# location title和blurb的取得长度值
maxPoolSize = 40
LOCATION_LENTGH, TITLE_LENGTH, BLURB_LENGTH = 3, 15, 200

class DataLoad:
    def __init__(self):
        '''
        books_with_blurbs.csv cloumns: ISBN,text,Author,Year,Publisher,Blurb
        BX-Book-Ratings.csv cloumns: User-ID,ISBN,Book-Rating
        BX-Books.csv cloumns: ISBN,Book-text,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L
        BX-Users.csv cloumns: User-ID,Location,Age
        '''
        # 线程池的任务列表
        self.task = []
        # 各个数据集
        self.BX_Users = self.load_origin('BX-Users')
        self.BX_Book_Ratings = self.load_origin('BX-Book-Ratings')
        self.Books = self.load_origin('books_with_blurbs', ',')
        
        #合并三个表
        self.features, self.ISBN2int, self.UserID2int, self.Users, self.blurb2vect, self.blurb2int = self.get_features()
        self.labels = self.features.pop('Book-Rating')

    def load_origin(self, 
        filename: "根据文件名获取源文件，获取正确得columns、values等值", 
        sep: "因为源文件的分隔方式sep不同，所以通过传参改编分隔方式"="\";\"", 
        )->pd.DataFrame:
        '''
        获取原始数据，第一遍获取后将用pickle保存到本地，方便日后调用
        '''

        try:
            # 从缓存的文件夹FILTERED_DATA_DIR获取基本被过滤后的文件
            pickled_data = pickle.load(open(FILTERED_DATA_DIR+filename+'.p', mode='rb'))
            return pickled_data
        except FileNotFoundError:
            # 如果缓存的文件不存在或者没有，则在源目录ORIGIN_DATA_DIR获取
            all_fearures = pd.read_csv(ORIGIN_DATA_DIR+filename+'.csv', engine='python',sep=sep, encoding='utf-8')
            # \";\"  初始过滤的文件
            # ,      初始不需要过滤的文件
            data_dict = {"\";\"":self.filtrator(all_fearures), ',':all_fearures}
            # 因为没获得处理后的文件，所以我们在获取源文件后可以保存一下处理后的文件
            pickle.dump((data_dict[sep]), open(FILTERED_DATA_DIR+filename+'.p', 'wb'))
            return data_dict[sep]
        except UnicodeDecodeError as e:
            ''' 测试时经常会出现编码错误，如果尝试更换编码方式无效，可以将编码错误的部分位置重新复制粘贴就可以了，这里我们都默认UTF-8'''
            print('UnicodeDecodeError:',e)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)


    def filtrator(self, 
        f_data: "输入需要进行初步filter的数据"
        )->pd.DataFrame:
        '''
        源文件中的columns和各个值得第一列的第一个字符和最后一列的最后一个字符都带有双引号‘"’,需要将其filter,Location字段当用户Age为null的时候，末尾会有\";NULL字符串 ，直接用切片调整
        '''
        Nonetype_age = 0
        f_data = f_data.rename(columns={f_data.columns[0]:f_data.columns[0][1:], f_data.columns[-1]:f_data.columns[-1][:-1]})
        f_data[f_data.columns[0]] = f_data[f_data.columns[0]].map(lambda v:v[1:] if v!=None else Nonetype_age)
        f_data[f_data.columns[-1]] = f_data[f_data.columns[-1]].map(lambda v:v[:-1] if v!=None else Nonetype_age)
        try:
            f_data = f_data[f_data['Location'].notnull()][f_data[f_data['Location'].notnull()]['Location'].str.contains('\";NULL')]
            f_data['Location'] = f_data['Location'].map(lambda location:location[:-6])
        except:
            pass
        return f_data

    def get_features(self):
        '''
        获取整个数据集的所有features，并对每个文本字段作xxxxx
        User-ID、Location、ISBN、Book-Rating、Title、Author、Year、Publisher、Blurb
        '''
        try:
            # 从缓存的文件夹FILTERED_DATA_DIR获取features的文件
            all_fearures, ISBN2int, UserID2int, Users, blurb2vect, blurb2int  = pickle.load(open(FILTERED_DATA_DIR+'features.p', mode='rb'))
            return all_fearures, ISBN2int, UserID2int, Users, blurb2vect, blurb2int 
        except:
            # 将所有的数据组成features大表
            all_fearures = pd.merge(pd.merge(self.BX_Users, self.BX_Book_Ratings), self.Books)
            Users = all_fearures
            
            executor = ThreadPoolExecutor(max_workers=maxPoolSize)
            # 因为没获得处理后的文件，所以我们在获取源文件后可以保存一下处理后的文件
            # isbn2index userid2index
            all_fearures.pop('Age')
            self.task = [
                executor.submit(self.word2int,all_fearures['ISBN']),
                executor.submit(self.word2int,all_fearures['Author']),
                executor.submit(self.word2int,all_fearures['Publisher']),
                executor.submit(self.word2int,all_fearures['Year']),
                executor.submit(self.word2int,all_fearures['User-ID']),
                executor.submit(self.list2int,all_fearures['Location'], LOCATION_LENTGH),
                executor.submit(self.text2int,all_fearures['Title'], TITLE_LENGTH),
                ]
            all_fearures['ISBN'], ISBN2int = self.task[0].result()
            all_fearures['Author'], X2int = self.task[1].result()
            all_fearures['Publisher'], X2int = self.task[2].result()
            all_fearures['Year'], X2int = self.task[3].result()
            all_fearures['User-ID'], UserID2int  = self.task[4].result()
            all_fearures['Location'] = self.task[5].result()
            all_fearures['Title'], x2vect, x2int = self.task[6].result()

            # all_fearures['ISBN'], ISBN2int = self.word2int(all_fearures['ISBN'])
            # all_fearures['Author'], X2int = self.word2int(all_fearures['Author'])
            # all_fearures['Publisher'], X2int = self.word2int(all_fearures['Publisher'])
            # all_fearures['Year'], X2int = self.word2int(all_fearures['Year'])
            # all_fearures['User-ID'], UserID2int  = self.word2int(all_fearures['User-ID'])
            # all_fearures['Location'] = self.list2int(all_fearures['Location'], LOCATION_LENTGH)
            # all_fearures['Title'] = self.text2int(all_fearures['Title'], TITLE_LENGTH)
            all_fearures['Blurb'], blurb2vect, blurb2int = self.text2int(all_fearures['Blurb'], BLURB_LENGTH)
            all_fearures['Book-Rating'] = all_fearures['Book-Rating'].astype('float32')
            pickle.dump((all_fearures, ISBN2int, UserID2int, Users, blurb2vect, blurb2int ), open(FILTERED_DATA_DIR+'features.p', 'wb'))
            return all_fearures, ISBN2int, UserID2int, Users, blurb2vect, blurb2int 

    @Config.logging_time
    def list2int(self,
    feature:'特征值', 
    length=0):
        cities = set()
        for val in feature.str.split(','):
            cities.update(val)
        city2int = {val:ii for ii, val in enumerate(cities)}
        list_map = {val:[city2int[row] for row in val.split(',')][:length] for ii,val in enumerate(set(feature))}
        return feature.map(list_map)

    @Config.logging_time
    def word2int(self,
    feature:'特征值'):
        word_map = {val:ii for ii,val in enumerate(set(feature))}
        return feature.map(word_map), word_map

    @Config.logging_time
    def text2int(self,
    feature:'特征值',
    length=0):
        '''
        将文本字段比如title、blurb只取英文单词，并用空格为分隔符，做成一个带index值的集合，并用index值表示各个单词，作为文本得表示
        '''
        # 过滤停用词,过滤前：长度 title 23815 blurb 127185 过滤后 title 23730 blurb 127034
        pattern = re.compile(r'[^a-zA-Z]')

        stop_words = [re.sub(pattern, ' ', _) for _ in nltk.corpus.stopwords.words('english')]

        filtered_map = {val:re.sub(pattern, ' ', val) for ii,val in enumerate(set(feature)) }
        letter_filter = lambda feature:feature.map({val:re.sub(pattern, ' ', str(val)) for ii,val in enumerate(set(feature)) })
        text_words = set()
        filtered_feature = letter_filter(feature)
        for val in filtered_feature.str.split(' '):
            # self.task.append(executor.submit(lambda val,stop_words:text_words.update(set(_ for _ in val if _ not in stop_words)),val,stop_words))
            text_words.update(set(_ for _ in val if _ not in stop_words))
        # 让text_words继续运行，主线程阻塞等待运行完成
        # wait(self.task, return_when=ALL_COMPLETED)
        text_words.add('<PAD>')
        text2int = {val:ii for ii, val in enumerate(text_words)}
        text2index = {val:[text2int[row] for row in filtered_map[val].split() if row in text2int][:length] for ii,val in enumerate(set(feature))}
        # word2vect 长度不截取，把过滤后的简介返回
        text2vect = {val:[row for row in filtered_map[val].split() if row in text2int] for ii,val in enumerate(set(feature))}

        for key in text2index:
            for cnt in range(length - len(text2index[key])):
                # self.task.append(executor.submit(lambda key,cnt:text2index[key].insert(len(text2index[key]) + cnt,text2int['<PAD>']),key,cnt))
                text2index[key].insert(len(text2index[key]) + cnt,text2int['<PAD>'])
        return feature.map(text2index), feature.map(text2vect), text2int

if __name__ == "__main__":
    test = DataLoad()
    print(test.blurb2vect)