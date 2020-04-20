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
