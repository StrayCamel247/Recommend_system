#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
try:
    from .Prepro_data import DataLoad
except ModuleNotFoundError:
    from Prepro_data import DataLoad

test = 1
origin_DATA = DataLoad()

# print('origin_DATA.features.head()\n', origin_DATA.Books.head())
# print('数据预处理完成')
# print('origin_DATA.features.head()\n', origin_DATA.features.head())

#User-ID number,max_index
all_user_id_number = len(set(origin_DATA.features['User-ID']))

#Location number index
location_length = len(origin_DATA.features['Location'][0])
all_location_words_number = max([j for i in origin_DATA.features['Location'] for j in i])+1

#ISBN numner
all_isbn_words_number = len(set(origin_DATA.features['ISBN']))

#Title
title_length = len(origin_DATA.features['Title'][0])
all_title_words_number = max([j for i in origin_DATA.features['Title'] for j in i])+1

#Author
all_author_words_number = len(set(origin_DATA.features['Author']))

#Year
all_year_words_number = len(set(origin_DATA.features['Year']))

# Publisher
all_publisher_words_number = len(set(origin_DATA.features['Publisher']))

# Blurb
blurb_length = len(origin_DATA.features['Blurb'][0])
all_blurb_words_number = max([j for i in origin_DATA.features['Blurb'] for j in i])+1

__all__ = [origin_DATA, all_user_id_number, location_length, all_location_words_number, all_isbn_words_number, all_title_words_number, title_length, all_author_words_number, all_year_words_number, all_publisher_words_number, blurb_length, all_blurb_words_number]

if __name__ == "__main__":
    print('blurb_length=%d,all_blurb_words_number=%d'% (blurb_length,all_blurb_words_number))
    print('all_publisher_words_number ',all_publisher_words_number)
    print('all_year_words_number ',all_year_words_number)
    print('all_author_words_number ',all_author_words_number)
    print('title_length=%d,all_title_words_number=%d'% (title_length,all_title_words_number))
    print('all_isbn_words_number ',all_isbn_words_number)
    print('location_length=%d,all_location_words_number=%d '% (location_length,all_location_words_number) )
    print('all_user_id_number=',all_user_id_number)
    # 主要是将数据和我们在Config中设置的是否一致，并去除之后将要用到的数据
    """
    blurb_length=200,all_blurb_words_number=127035
    all_publisher_words_number  2909
    all_year_words_number  81
    all_author_words_number  15196
    title_length=15,all_title_words_number=23731
    all_isbn_words_number  38036
    location_length=3,all_location_words_number=7574
    all_user_id_number= 28836
    """
    

