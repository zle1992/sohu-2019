import pandas as pd
import json
import keras
import numpy as np
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout,\
                         TimeDistributed, Concatenate, Dense, GRU, Conv1D,\
                         LeakyReLU,concatenate,GlobalMaxPool1D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
# from CrfLayer import CRF
from keras_contrib.layers import CRF


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)


import sys
sys.path.append('../common')
from util import *

def word2id(texts,MAXLEN):
    '''
    texts = [['我', '是', '谁', '啊', '啊', '啊', '啊'],
    ['我', '是', '谁', '啊', '啊', '啊', '啊']]
    '''

    tokenizer = Tokenizer(num_words=None,) # 分词MAX_NB_WORDS
    tokenizer.fit_on_texts(texts)
    X_train_word_ids = tokenizer.texts_to_sequences(texts) #受num_words影响
    x_data=pad_sequences(X_train_word_ids,maxlen=MAXLEN,truncating='post',padding='post',value=0)

    n_vocab_char = len(tokenizer.index_word)+1

    return x_data,n_vocab_char








class CRF_KERAS(object):
    """docstring for ClassName"""
    def __init__(self,):
        super(CRF_KERAS, self).__init__()
        self.agg = True
        self.parms ={
       # 'n_vocab_char':0,
        'MAXLEN':512,
        'n_entity' :4,
        'embed_size' : 256,
        'n_lstm' :256,
        'epochs':5,
        'batch_size':512,
        }


    def preprocessing(self,df_train,df_test):
        self.train_size = len(df_train)
        df_new = df_train.append(df_test)
        df_new = df_new.reset_index()
        df_new['labels']=df_new.apply(lambda x:text2id(x,col='texts'),axis=1)
        df_new['texts'] = df_new['texts'].map(lambda x:list(x))
        texts=df_new['texts'].values
        texts_id ,self.parms['n_vocab_char']= word2id(texts,self.parms['MAXLEN'])

        labels = df_new['labels'].values
        labels_id=pad_sequences(labels,maxlen=self.parms['MAXLEN'],truncating='post',padding='post',value=0)

        print('labels_id shape',labels_id.shape)
        print('texts_id',texts_id.shape)
        self.X_train ,self.X_test = texts_id[:self.train_size],texts_id[self.train_size:]
        self.Y_train ,self.Y_test = labels_id[:self.train_size],labels_id[self.train_size:]

    def get_model(self):
      # main
        char_input = Input(shape=(None,), name='main_input')
        x = Embedding(input_dim=self.parms['n_vocab_char'],
                      output_dim=self.parms['embed_size'],
                      #mask_zero=True,
                      trainable=True)(char_input)
        x = Bidirectional(CuDNNLSTM(self.parms['n_lstm'], return_sequences=True))(x)
        crf=CRF(self.parms['n_entity'], sparse_target=True)
        output = crf(x)

        model = Model(inputs=[char_input],
                           outputs=output)
        model.compile(optimizer="adam",
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])
        model.summary()
        return model
    def train(self,):
       
       
        x_train,x_dev,y_train,y_dev=train_test_split(self.X_train,self.Y_train ,test_size=0.001, random_state=42)
        print('y_train',y_train.shape)
        y_train=np.expand_dims(y_train, 2) 
        y_dev=np.expand_dims(y_dev, 2) 
        self.model = self.get_model()
        K.set_value(self.model.optimizer.lr, 0.01)
        self.model.fit(x_train,y_train,epochs=self.parms['epochs'],batch_size=self.parms['batch_size'],validation_data=(x_dev,y_dev))
    def predict(self,df):
                
        texts=df['texts'].values
        y = np.argmax(self.model.predict(self.X_test),axis=2)
        final_res = decoder(y,texts)
        df['entity_pred']= final_res
        return df

    

class CRF_BERT(CRF_KERAS):
    """docstring for ClassName"""
    def __init__(self,):
        super(CRF_BERT, self).__init__()
        self.agg = True
        self.parms ={
       # 'n_vocab_char':0,
        'MAXLEN':768,
        'n_entity' :4,
        'embed_size' : 256,
        'n_lstm' :256,
        'epochs':5,
        'batch_size':512,
        }
        from kashgari.embeddings import BERTEmbedding
        model_path="/sdb1/zhangle/2019/zhangle11/bert/chinese_L-12_H-768_A-12"

        self.embedding = BERTEmbedding(model_path, 512)
    def text2id(self,texts):
        train_x_id = []
        for text in texts:
            t_id = []
            for char in text:
                if char in  self.embedding.token2idx:
                    t_id.append(self.embedding.token2idx[char])
                else:
                    t_id.append(self.embedding.token2idx['[UNK]'])

            train_x_id.append(t_id)
        return train_x_id

    def preprocessing(self,df_train,df_test):
        
        
        
        
        self.train_size = len(df_train)
        df_new = df_train.append(df_test)
        df_new = df_new.reset_index()
        df_new['labels']=df_new.apply(lambda x:text2id(x,col='texts'),axis=1)
        df_new['texts'] = df_new['texts'].map(lambda x:list(x))
        
        

        x = df_new['texts'].values
        x = self.text2id(x)
        y = df_new['labels'].values
        x= [ i for i in x]
        y= [ i for i in y]
        
        texts_id=pad_sequences(x,maxlen=self.parms['MAXLEN'],truncating='post',padding='post',value=0)
        labels_id=pad_sequences(y,maxlen=self.parms['MAXLEN'],truncating='post',padding='post',value=0)
        
        
        

        print('labels_id shape',labels_id.shape)
        print('texts_id',texts_id.shape)
        self.X_train ,self.X_test = texts_id[:self.train_size],texts_id[self.train_size:]
        self.Y_train ,self.Y_test = labels_id[:self.train_size],labels_id[self.train_size:]

    def get_model(self):
        
      # main
        #char_input = Input(shape=(None,), name='main_input')
        x = self.embedding.model.output
        x = Bidirectional(CuDNNLSTM(self.parms['n_lstm'], return_sequences=True))(x)
        crf=CRF(self.parms['n_entity'], sparse_target=True)
        output = crf(x)

        model = Model(inputs=[self.embedding.model.inputs],
                           outputs=output)
        model.compile(optimizer="adam",
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])
        model.summary()
        return model



class BaseTrain(object):
    """docstring for BaseTrain"""
    def __init__(self, Entity_MODEL):
        super(BaseTrain, self).__init__()
        self.train_path='../data/coreEntityEmotion_train.txt.agg.pick'
        self.test_path = '../data/coreEntityEmotion_test_stage1.txt.agg.pick'
        self.agg = False
        self.Entity_MODEL = Entity_MODEL
        self.read_data()#初始化的时候必须要有read_data


    def read_data(self):
        print('read_data ing')
        self.df_train = pd.read_pickle(self.train_path).head(10)
        self.df_test = pd.read_pickle(self.test_path).head(10)
      
      

        print(list(self.df_train))
        print('read_data done')
    def train_Entity(self):
        self.Entity_MODEL.preprocessing(self.df_train,self.df_test)
        self.Entity_MODEL.train()
    def predict_Entity(self,df_test):
        
        df_test = self.Entity_MODEL.predict(df_test)
        
        df_agg=postposs(df_test)
        df_agg['entity_pred'] = df_agg['entity_pred'].map(lambda x:post(x))
        #df_val['texts'] = df_val['texts'].map(lambda x:' '.join(x))
        return df_agg[['entity','newsId','entity_pred']]
    
    def predict_Emotion(self,df_test):
        
        df_test['emotion_pred']=df_test.apply(lambda x:['POS']*len(x['entity_pred']),axis=1)
        #df_test['emotion_pred']=df_test.apply(lambda x:['POS']*3,axis=1)

        return df_test
    def submit(self):
        df_test = self.predict_Entity(self.df_test)
        
        
        df_test['entity_pred']= df_test['entity_pred'].map(lambda x:['杨超越'] if len(x)==0 else x)
        df_test = self.predict_Emotion(df_test)

        df_test['entity_pred'] = df_test['entity_pred'].map(lambda x:','.join(x))
        df_test['emotion_pred'] = df_test['emotion_pred'].map(lambda x:','.join(x))



        path = '/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_sample_submission_v2.txt'
        df1 =pd.read_csv(path,names=['newsId','entity','emotion'],sep='\t')

        df_test = pd.merge(df1[['newsId']],df_test[['newsId','entity_pred','emotion_pred']],how='left')
        df_test[['newsId','entity_pred','emotion_pred']].to_csv('keras_res.txt', sep='\t',index=False,header=False)


def main():
    Entity_MODEL = CRF_KERAS()
    #Entity_MODEL = CRF_BERT()

    Train  = BaseTrain(Entity_MODEL)
    Train.train_Entity()
    Train.submit()

if __name__ == '__main__':
  main()

