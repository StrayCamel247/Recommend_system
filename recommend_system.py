import os, sys, re, pickle
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
print(sys.version_info)
for module in tf, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
ORIGIN_DATA_DIR = os.getcwd()+'/data/BX-CSV-Dump/'
FILTERED_DATA_DIR = os.getcwd()+'/tmp/'
class DataLoad:
    def __init__(self):
        '''
        books_with_blurbs.csv cloumns: ISBN,Title,Author,Year,Publisher,Blurb
        BX-Book-Ratings.csv cloumns: User-ID,ISBN,Book-Rating
        BX-Books.csv cloumns: ISBN,Book-Title,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L
        BX-Users.csv cloumns: User-ID,Location,Age
        '''
        self.BX_Users = self.load_origin('BX-Users')
        self.BX_Book_Ratings = self.load_origin('BX-Book-Ratings')
        self.books_with_blurbs = self.load_origin('books_with_blurbs', ',')
        #合并三个表
        self.features = self.get_features()


    def get_features(self):
        try:
            # 从缓存的文件夹FILTERED_DATA_DIR获取features的文件
            pickled_data = pickle.load(open(FILTERED_DATA_DIR+'features.p', mode='rb'))
            return pickled_data
        except FileNotFoundError:
            # 将所有的数据组成features大表
            data = pd.merge(pd.merge(self.BX_Users, self.BX_Book_Ratings), self.books_with_blurbs)
            # 因为没获得处理后的文件，所以我们在获取源文件后可以保存一下处理后的文件
            pickle.dump(data, open(FILTERED_DATA_DIR+'features.p', 'wb'))
            return data


    def filtrator(self,f_data):
        '''
        源文件中的columns和各个值得第一列的第一个字符和最后一列的最后一个字符都带有双引号‘"’,需要将其filter
        数据中值为NULL，我们将其替换成'<PAD>'
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


    def load_origin(self,
        filename: "根据文件名获取源文件，获取正确得columns、values等值",
        sep: "因为源文件的分隔方式sep不同，所以通过传参改编分隔方式"="\";\"",
        )->pd.DataFrame:
        try:
            # 从缓存的文件夹FILTERED_DATA_DIR获取基本被过滤后的文件
            with open(FILTERED_DATA_DIR+filename+'.p','rb')as f:
                pickled_data = pickle.load(f)
                return pickled_data
        except FileNotFoundError:
            # 如果缓存的文件不存在或者没有，则在源目录ORIGIN_DATA_DIR获取
            data = pd.read_csv(ORIGIN_DATA_DIR+filename+'.csv', engine='python',sep=sep, encoding='utf-8')
            # \";\"  初始过滤的文件
            # ,      初始不需要过滤的文件
            print(data.head())
            data_dict = {"\";\"":self.filtrator(data), ',':data}
            # 因为没获得处理后的文件，所以我们在获取源文件后可以保存一下处理后的文件
            pickle.dump((data_dict[sep]), open(FILTERED_DATA_DIR+filename+'.p', 'wb'))
            return data_dict[sep]
        except UnicodeDecodeError as e:
            ''' 测试时经常会出现编码错误，如果尝试更换编码方式无效，可以将编码错误的部分位置重新复制粘贴就可以了，这里我们都默认UTF-8'''
            print('UnicodeDecodeError:',e)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)


    def __del__(self):
        pass

if __name__=='__main__':
    origin_DATA = DataLoad()
    print(origin_DATA.features.head())
    print(origin_DATA.BX_Users.head())
    print(origin_DATA.BX_Book_Ratings.head())
    print(origin_DATA.books_with_blurbs.head())
