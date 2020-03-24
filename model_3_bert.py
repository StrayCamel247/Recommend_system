import os, sys, re, pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
import math
from tensorflow import keras
from transformers import BertTokenizer, BertConfig, TFBertModel
from tqdm import tqdm
print(sys.version_info)

for module in tf, mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
#  预训练模型来自于 https://github.com/huggingface/transformers
ORIGIN_DATA_DIR = os.getcwd() + '/all_fearures/BX-CSV-Dump/'
FILTERED_DATA_DIR = os.getcwd() + '/tmp_bert/'
#
ORIGIN_DATA_DIR = os.getcwd() + '/all_fearures/BX-CSV-Dump/'
FILTERED_DATA_DIR = os.getcwd() + '/tmp_bert/'


class DataLoad:
    def __init__(self):
        '''
        books_with_blurbs.csv cloumns: ISBN,text,Author,Year,Publisher,Blurb
        BX-Book-Ratings.csv cloumns: User-ID,ISBN,Book-Rating
        BX-Books.csv cloumns: ISBN,Book-text,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L
        BX-Users.csv cloumns: User-ID,Location,Age
        '''
        self.BX_Users = self.load_origin('BX-Users')
        self.BX_Book_Ratings = self.load_origin('BX-Book-Ratings')
        self.books_with_blurbs = self.load_origin('books_with_blurbs', ',')
        # 合并三个表
        self.features = self.get_features()
        self.labels = self.features.pop('Book-Rating')

    def load_origin(self,
                    filename: "根据文件名获取源文件, 获取正确得columns、values等值",
                    sep: "因为源文件的分隔方式sep不同, 所以通过传参改编分隔方式" = "\";\"",
                    ) -> pd.DataFrame:
        '''
        获取原始数据，第一遍获取后将用pickle保存到本地，方便日后调用
        '''
        try:
            # 从缓存的文件夹FILTERED_DATA_DIR获取基本被过滤后的文件
            pickled_data = pickle.load(open(FILTERED_DATA_DIR + filename + '.p', mode='rb'))
            return pickled_data
        except FileNotFoundError:
            # 如果缓存的文件不存在或者没有，则在源目录ORIGIN_DATA_DIR获取
            all_fearures = pd.read_csv(ORIGIN_DATA_DIR + filename + '.csv', engine='python', sep=sep, encoding='utf-8')
            # \";\"  初始过滤的文件
            # ,      初始不需要过滤的文件
            data_dict = {"\";\"": self.filtrator(all_fearures), ',': all_fearures}
            # 因为没获得处理后的文件，所以我们在获取源文件后可以保存一下处理后的文件
            pickle.dump((data_dict[sep]), open(FILTERED_DATA_DIR + filename + '.p', 'wb'))
            return data_dict[sep]
        except UnicodeDecodeError as e:
            ''' 测试时经常会出现编码错误，如果尝试更换编码方式无效，可以将编码错误的部分位置重新复制粘贴就可以了，这里我们都默认UTF-8'''
            print('UnicodeDecodeError:', e)
        except pd.errors.ParserError as e:
            print("connect error|pandas Error: %s" % e)

    def filtrator(self,
                  f_data: "输入需要进行初步filter的数据"
                  ) -> pd.DataFrame:
        '''
        源文件中的columns和各个值得第一列的第一个字符和最后一列的最后一个字符都带有双引号‘"’,需要将其filter,Location字段当用户Age为null的时候，末尾会有\";NULL字符串 ，直接用切片调整
        '''
        Nonetype_age = 0
        f_data = f_data.rename(
            columns={f_data.columns[0]: f_data.columns[0][1:], f_data.columns[-1]: f_data.columns[-1][:-1]})
        f_data[f_data.columns[0]] = f_data[f_data.columns[0]].map(lambda v: v[1:] if v != None else Nonetype_age)
        f_data[f_data.columns[-1]] = f_data[f_data.columns[-1]].map(lambda v: v[:-1] if v != None else Nonetype_age)
        try:
            f_data = f_data[f_data['Location'].notnull()][
                f_data[f_data['Location'].notnull()]['Location'].str.contains('\";NULL')]
            f_data['Location'] = f_data['Location'].map(lambda location: location[:-6])
        except:
            pass
        return f_data

    def get_features(self):
        '''
        获取整个数据集的所有features，并对每个文本字段作xxxxx
        User-ID
        Location
        ISBN
        Book-Rating
        Title
        Author
        Year
        Publisher
        Blurb
        '''
        try:
            # 从缓存的文件夹FILTERED_DATA_DIR获取features的文件
            pickled_data = pickle.load(open(FILTERED_DATA_DIR + 'features.p', mode='rb'))
            return pickled_data
        except FileNotFoundError:
            # 将所有的数据组成features大表
            all_fearures = pd.merge(pd.merge(self.BX_Users, self.BX_Book_Ratings), self.books_with_blurbs)
            # 因为没获得处理后的文件，所以我们在获取源文件后可以保存一下处理后的文件
            all_fearures.pop('Age')
            all_fearures['Title'] = self.feature2int(all_fearures['Title'], 'text', 15)
            all_fearures['Blurb'] = self.feature2int(all_fearures['Blurb'], 'text', 200)
            all_fearures['ISBN'] = self.feature2int(all_fearures['ISBN'], 'word')
            all_fearures['Author'] = self.feature2int(all_fearures['Author'], 'word')
            all_fearures['Publisher'] = self.feature2int(all_fearures['Publisher'], 'word')
            all_fearures['User-ID'] = self.feature2int(all_fearures['User-ID'], 'word')
            all_fearures['Year'] = self.feature2int(all_fearures['Year'], 'word')
            all_fearures['Location'] = self.feature2int(all_fearures['Location'], 'list')
            all_fearures['Book-Rating'] = all_fearures['Book-Rating'].astype('float32')
            pickle.dump(all_fearures, open(FILTERED_DATA_DIR + 'features.p', 'wb'))
            return all_fearures

    def feature2int(self,
                    feature: '特征值',
                    feature_type: 'text/word/list',
                    length: '文本设置的最大长度' = 0,
                    ):
        '''
        将文本字段比如title、blurb只取英文单词，并用空格为分隔符，做成一个带index值的集合，并用index值表示各个单词，作为文本得表示
        '''
        pattern = re.compile(r'[^a-zA-Z]')
        filtered_map = {val: re.sub(pattern, ' ', str(val)) for ii, val in enumerate(set(feature))}

        word_map = {val: ii for ii, val in enumerate(set(feature))}

        try:
            cities = set()
            for val in feature.str.split(','):
                cities.update(val)
            city_index = {val: ii for ii, val in enumerate(cities)}
            list_map = {val: [city_index[row] for row in val.split(',')][:3] for ii, val in enumerate(set(feature))}
        except AttributeError:
            list_map = {}

        feature_dict = {
            'text': feature.map(filtered_map),
            'word': feature.map(word_map),
            'list': feature.map(list_map),
        }
        return feature_dict[feature_type]

    def __del__(self):
        pass


origin_DATA = DataLoad()
max_sequence_length = 512
bert_path = './bert_models/'
# title+blurb的最大长度
blurb_series = origin_DATA.features.Blurb
title_series = origin_DATA.features.Title

class pre_title_blurb():
    def __init__(self, lengths):
        self.max_sequence_length = lengths
        self.bert_path = './bert_models/'
        self.blurb_series = origin_DATA.features.Blurb
        self.title_series = origin_DATA.features.Title
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path+'bert-base-uncased-vocab.txt')
        self.input_ids, self.input_type_ids, self.input_masks = self.return_id()
    def return_id(self):
        input_ids, input_type_ids, input_masks = [], [], []
        for index in tqdm(range(math.ceil(blurb_series.shape[0]/10))):
            # 这个encode_plus可以帮助我们自动标记text,pos,segement的token
            inputs = self.tokenizer.encode_plus(text=title_series[index], text_pair=blurb_series[index], add_special_tokens=True,
                                           max_length=self.max_sequence_length, truncation_strategy='longest_first')
            # 我们需要手动补齐,得到三个向量对应的token
            padding_id = self.tokenizer.pad_token_id
            input_id = inputs['input_ids']
            padding_length = self.max_sequence_length-len(input_id)
            input_id = inputs['input_ids'] + [padding_id] * (padding_length)
            input_type_id = inputs['token_type_ids']
            input_type_id = input_type_id + [0] * padding_length
            input_mask = inputs['attention_mask']
            input_mask = input_mask + [0] * padding_length
            input_ids.append(input_id)
            input_type_ids.append(input_type_id)
            input_masks.append(input_mask)
        return np.array(input_ids), np.array(input_type_ids), np.array(input_masks)

data_blurb_title = pre_title_blurb(lengths=max_sequence_length)

print(data_blurb_title.input_ids.shape)
print(data_blurb_title.input_masks.shape)
print(data_blurb_title.input_type_ids.shape)

# user-id的字典,总共有28836个用户
all_user = len(set(origin_DATA.features['User-ID']))
new_user_id = {val: i for i, val in enumerate(set(origin_DATA.features['User-ID']))}
print('all user id = ', all_user)
# location的数量=7573(从0开始的)
all_location = max([j for i in origin_DATA.features.Location for j in i]) +1
print('all location = ', all_location)

# ISBN总数
all_isbn = len(set(origin_DATA.features['ISBN']))
print('all isbn = ', all_isbn)
# author总数
all_author = len(set(origin_DATA.features['Author']))
print('all author = ', all_author)
# year总数
all_year = len(set(origin_DATA.features['Year']))
print('all year = ', all_year)
# publish总数
all_publisher = len(set(origin_DATA.features['Publisher']))
print('all publisher = ', all_publisher)


def get_inputs():
    # 用户特征输入
    user_id = keras.layers.Input(shape=(1,), dtype='int32', name='user_id_input')
    user_location = keras.layers.Input(shape=(3,), dtype='int32', name='user_location_input')

    book_title_blurb_id = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name='book_title_blurb_id')
    book_title_blurb_type_id = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name='book_title_blurb_type_id')
    book_title_blurb_mask = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name='book_title_blurb_mask')
    return user_id, user_location,  book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask

# 嵌入矩阵的维度
embed_dim = 8
embed_dim_words = 16

def user_embed_layer(u_id, u_loca):
    user_id_embedd = keras.layers.Embedding(all_user, embed_dim, name='user_id_embedding')(u_id)
    user_loca_embedd = keras.layers.Embedding(all_location, embed_dim , name='user_loca_embedding')(u_loca)
    return user_id_embedd, user_loca_embedd

def get_user_feature(u_id_embedd, u_loca_embedd):
    u_id_layer = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='u_id_dense')(u_id_embedd)
#     u_id_layer_drop = keras.layers.Dropout(rate=0.5, name='u_id_layer_drop')(u_id_layer)
    # u_id_layer.shape = (?, 1, 64)
    # u_loca_layer.shape = (?, 64)
    # 这里可以再加个Dense
    u_loca_layer = keras.layers.LSTM(16, go_backwards=True, name='u_loca_lstm')(u_loca_embedd)
    u_loca_layer_lstm = keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.nn.l2_loss, name='u_loca_layer_lstm')(u_loca_layer)
    u_id_reshape = keras.layers.Reshape([32])(u_id_layer)
    u_combine = keras.layers.concatenate([u_id_reshape, u_loca_layer_lstm],axis=1, name='u_combine')
    print(u_combine.shape)
    # 这里能不能用激活函数
    u_feature_layer = keras.layers.Dense(100, activation='tanh', name='u_feature_layer')(u_combine)
    print(u_feature_layer.shape)
    return u_feature_layer


def get_book_feature(book_title_id, book_title_type_id, book_title_mask):
    config = BertConfig()
    # 获取隐藏层的信息
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(bert_path + 'bert-base-uncased-tf_model.h5', config=config)
    book_title_cls = bert_model(book_title_id, attention_mask=book_title_mask, token_type_ids=book_title_type_id)
    print(len(book_title_cls))
    print(book_title_cls[0].shape)
    print(book_title_cls[1].shape)
    book_feature_layer = keras.layers.Dense(100, activation='tanh')(book_title_cls[1])
    return book_feature_layer


def get_rating(user_feature, book_feature):
#     multiply_layer = keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0]+layer[1], axis=1, keepdims=True), name = 'user_book_feature')((user_feature, book_feature))
    inference_layer = keras.layers.concatenate([user_feature, book_feature], axis=1, name='user_book_feature')
    inference_dense = tf.keras.layers.Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(inference_layer)
    multiply_layer = tf.keras.layers.Dense(1, name="inference")(inference_layer)  # inference_dense
    return multiply_layer

def creat_model():
    user_id, user_location, book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask = get_inputs()
    user_id_embedd, user_loca_embedd = user_embed_layer(user_id, user_location)
    u_feature_layer = get_user_feature(user_id_embedd, user_loca_embedd)
    book_feature_layer = get_book_feature(book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask)
    multiply_layer = get_rating(u_feature_layer, book_feature_layer)
    model = keras.Model(inputs=[user_id, user_location, book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask],
                        outputs=[multiply_layer])
    return model

model = creat_model()
print(model.summary())

def get_train_val_test():
    m = len(origin_DATA.features['Location'])
    m = math.ceil(m/10)
    # 对location取3位数
    loca = np.zeros((m, 3))
    for i in range(m):
        loca[i] = np.array(origin_DATA.features['Location'][i])

    print(loca[:-2])
    input_features = [origin_DATA.features['User-ID'].to_numpy(), loca,
                     data_blurb_title.input_ids, data_blurb_title.input_masks, data_blurb_title.input_type_ids]
    labels = origin_DATA.labels.to_numpy()
    # 分割数据集以及shuffle
    np.random.seed(100)
    number_features = len(input_features)
    shuffle_index = np.random.permutation(m)
    shuffle_train_index = shuffle_index[:math.ceil(m*0.96)]
    shuffle_val_index = shuffle_index[math.ceil(m*0.96): math.ceil(m*0.98)]
    shuffle_test_index = shuffle_index[math.ceil(m*0.98):]
    train_features = [input_features[i][shuffle_train_index] for i in range(number_features)]
    train_labels = labels[shuffle_train_index]
    val_features = [input_features[i][shuffle_val_index] for i in range(number_features)]
    val_lables = labels[shuffle_val_index]
    test_features = [input_features[i][shuffle_test_index] for i in range(number_features)]
    test_lables = labels[shuffle_test_index]
    return train_features, train_labels, val_features, val_lables, test_features, test_lables

train_features, train_labels, val_features, val_lables, test_features, test_lables = get_train_val_test()
print(train_features[0].shape)
print(val_features[0].shape)
print(test_features[0].shape)

class model_network():
    def __init__(self):
        self.batchsize = 2
        self.epoch = 3
    def creat_model(self):
        user_id, user_location,  book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask = get_inputs()
        user_id_embedd, user_loca_embedd = user_embed_layer(user_id, user_location)
        u_feature_layer = get_user_feature(user_id_embedd, user_loca_embedd)
        b_feature_layer = get_book_feature(book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask)
        multiply_layer = get_rating(u_feature_layer, b_feature_layer)
        model = keras.Model(inputs=[user_id, user_location,  book_title_blurb_id, book_title_blurb_type_id, book_title_blurb_mask] ,
                    outputs=[multiply_layer])
        return model
    def train_model(self):
        model_optimizer = keras.optimizers.Adam()
        model = self.creat_model()
        model.compile(optimizer=model_optimizer, loss=keras.losses.mse)
        history = model.fit(train_features, train_labels, validation_data=(val_features, val_lables), epochs=self.epoch, batch_size=self.batchsize, verbose=1)
        return model, history
    def predict_model(self, model):
        test_loss = model.evaluate(test_features, test_lables, batch_size=self.batchsize, verbose=1)
        return test_loss

net_work = model_network()
model, history = net_work.train_model()