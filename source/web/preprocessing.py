# -*- coding:utf-8 -*-
import pandas as pd
import numpy  as np
import os


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
Book = book.drop(['bookmarkID'], axis=1)
Book.replace(np.nan, 0, inplace=True)
Book.replace(np.inf, 0, inplace=True)
Book['tagID_int']=Book['tagID'].astype(int)
Book['tagWeight_int']=Book['tagWeight'].astype(int)
Book = Book.drop(['tagID', 'tagWeight'], axis=1)
Book = Book.rename(columns={'tagID_int':'tagID', 'tagWeight_int':'tagWeight'})
# Book.to_csv('book.zip', index=False, compression=dict(method='zip', archive_name='book.csv'))

user = pd.merge(user_contacts, user_taggedbookmarks, on='userID', how='left')
User = user.rename(columns={'bookmarkID': 'BookID'})
# User.to_csv('user.zip', index=False, compression=dict(method='zip', archive_name='user.csv'))

Tag = tags.astype({'value':str})
# Tag.to_csv('tag.zip', index=False, compression=dict(method='zip', archive_name='tag.csv'))


print(Book)
print(User)
print(Tag)

