## Abstract
This is a simple Recommendation webapp using django web framework.

## Description
This is a simple Recommendation webapp using django web framework.

## 数据预处理
> 原始数据
- BX-Books.csv cloumns: 
    - ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, Image-URL-L

- books_with_blurbs.csv cloumns: 
    - ISBN, Title, Author, Year, Publisher, Blurb
    
- BX-Book-Ratings.csv cloumns: 
    - User-ID, ISBN, Book-Rating

- BX-Users.csv cloumns: 
    - User-ID, Location, Age

### 合成后
Features:
- BX-Book-Ratings（用户-对书籍的评分表）中，删除没有blurb的书籍对应的行的信息
    - User-ID, Location, Age
- 用 books_with_blurbs.csv 作为基本的表，所有数据都包括
    - ISBN, Title, Author, Year, Publisher, Blurb
- 从BX-Book-Ratings中提取相应的评分
    - Book-Rating