# 推荐系统
集合了协同过滤基础推荐算法框架、nlp自然语言处理分析文本技术，利用tf2、blert工具构建，django/flask可视化

## Contributors
🤟💪✍🏄‍♂️结识大家，三生有幸，have a bright future fo you。
- 🤸‍♀️[Stray_Camel](https://github.com/Freen247)
- 🕵️‍♂️[shange1996](https://github.com/shange1996)
- 👩‍🎓[lyj1998](https://github.com/lyj1998)
- 👩‍💻[EvaGalois](https://github.com/EvaGalois)

## Model_0
- 对user
    - user_id：Embedding-Dense/activation='relu'-concatenate0-Dense-Reshape-user_dense_layer_flat
user_dense_layer_flat
    - user_location：Embedding-Dense/activation='relu'-GRU-concatenate0-Dense-Reshape-user_dense_layer_flat
user_dense_layer_flat
- 对item：
    - book_isbn：Embedding-Dense/activation='relu'-concatenate1
    - book_author：Embedding-Dense/activation='relu'-concatenate1
    - book_year：Embedding-Dense/activation='relu'-concatenate1
    - book_publisher：Embedding-Dense/activation='relu'-concatenate1
    - book_title：Embedding-Dense/activation='relu'-Conv2D（多个窗口）-MaxPooling2D-concatenate0（多次池化后的结果）-Reshape-Dropout-concatenate1
    - book_blurb：Embedding-Dense/activation='relu'-GRU-concatenate1

## Model_1
## Model_2
## Model_3
## Model_4

## 文件目录概览
F:\recommend_system>tree /F
- │  .gitignore                           git上传时自动忽略的文件         
- │  model_0.ipynb                        
- │  model_1.ipynb
- │  model_1.png                          模型的架构图
- │  model_2.ipynb
- │  model_2.png                          模型的架构图
- │  model_3_bert.py                       s
- │  常见面试问题收集.ipynb                关于机器学习、nlp、推荐系统等面试问题的手机
- │
- ├─Data
- │  │  bags_of_title.csv
- │  │  books.csv
- │  │  BX-CSV-Dump.zip                   当前使用的数据集，解压后得到BX-CSV-Dump文件夹，运行项目时请先解压
- │  │  hetrec2011-delicious-2k.zip       之前用过的数据集1，数据集不够好
- │  │  Prepro_data.ipynb                    数据预处理后的展示
- │  │  Prepro_data.py                       对模型所有需要的数据进行预处理，并封装成对象
- │  │  tags.csv                          之前用过的数据集1，数据集不够好
- │  │  users.csv                         之前用过的数据集1，数据集不够好
- │  │  __init__.py
- │
- ├─models
    - 模型models缓存的文件
- │
- ├─Public
    - 模型超参，等工具函数封装成public对象方便往后调用
- │  │  Config.py
- │  │  __init__.py
- │  │
- │  └─__pycache__
- │          Config.cpython-36.pyc
- │          __init__.cpython-36.pyc
- │
- ├─source
- │  │  db.sqlite3                      django自带数据库
- │  │  manage.py                       django启动入口
- │  │
- │--- ├─main                           django的配置文件
- │--- ├─templates                      前端html文件
- │  │
- │  └─web                              利用django对模型进行可视化
- │          CF_CB.py                   协同过滤，用np、pd，实现基层理论，没有用tf
- │          mf.py                      矩阵分解，用np、pd，实现基层理论，没有用tf
- │          op_api.py                  数据预处理
- │          xxx.py                     django的组件文件
- │
- └─Tmp
    -  第一次数据预处理后用pickle缓存在这个文件夹，之后的调用数据直接从这个文件夹调用