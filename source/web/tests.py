
#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
import pandas as pd
import numpy as np

# fix error : _csv.Error: field larger than field limit (131072)
import csv
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class CF_CB():
    '''
    Collaborative Filtering + Content-based Filtering
    '''
    def __init__(self):
        pass
    
    def Matrix_factorization(self):
        book_tag = {'book':[],'given_tag':[]}
        books = []
        tags = []
        user_s = pd.read_csv(os.path.dirname(__file__)+'/data/users.csv', nrows=5, sep='\t', engine='python', dtype={'history':dict})
        for _ in user_s['history']:
            try:
                books.extend(set(_['book']))
                tags.extend(set(_['given_tag']))
                book_tag.append(_)
            except TypeError as e:
                books.extend(set(eval(_)['book']))
                tags.extend(set(eval(_)['given_tag']))
                book_tag['book']+=eval(_)['book']
                book_tag['given_tag']+=eval(_)['given_tag']
        books.sort()
        books = list(set(books))
        tags.sort()
        tags = list(set(tags))
        users = list(user_s['user'])

        # UBdata: index=users,columns=books
        UBdata = [[eval(list(user_s.loc[user_s['user'] == u,'history'])[0])['book'].count(int(b)) if b in eval(list(user_s.loc[user_s['user'] == u,'history'])[0])['book']  else 0 for b in books] for u in users]
        user_books = pd.DataFrame(UBdata,index=users,columns=books)

        # UTdata: index=users,columns=tags
        UTdata = [[eval(list(user_s.loc[user_s['user'] == u,'history'])[0])['given_tag'].count(int(t)) if t in eval(list(user_s.loc[user_s['user'] == u,'history'])[0])['given_tag']  else 0 for t in tags] for u in users]
        users_tags = pd.DataFrame(UTdata,index=users,columns=tags)

        # TBdata: index=books,columns=tags
        TBdata = np.zeros((len(tags),len(books)))
        for b_i,b_v in enumerate(book_tag['book']):
            t_v=book_tag['given_tag'][b_i]
            TBdata[tags.index(t_v),books.index(b_v)] +=1
        tags_books = pd.DataFrame(TBdata,index=tags,columns=books)
        # user_books.to_csv(os.path.dirname(__file__)+'/tmp/users_books.csv', sep='\t', header=True, index=True)
        Matrix_factorization_users_books = pd.DataFrame(users_tags.values.dot(tags_books.values),index=users,columns=books)
        Matrix_factorization_users_books.to_csv(os.path.dirname(__file__)+'/tmp/Matrix_factorization_users_books.csv', sep='\t', header=True, index=True)
        print('矩阵分解~~')


if __name__ == "__main__":
    test = CF_CB()
    test.Matrix_factorization()
    