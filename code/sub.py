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

# from CrfLayer import CRF
from keras_contrib.layers import CRF


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)


from read import read_data


        

MAXLEN=50
n_entity =4
embed_size = 256
n_lstm =256


def word2id(texts):


  tokenizer = Tokenizer(num_words=None,) # 分词MAX_NB_WORDS
  tokenizer.fit_on_texts(texts)
  X_train_word_ids = tokenizer.texts_to_sequences(texts) #受num_words影响
  x_data=pad_sequences(X_train_word_ids,maxlen=MAXLEN,truncating='post',padding='post',value=0)

  n_vocab_char = len(tokenizer.index_word)+1
 
  return x_data,n_vocab_char



def get_model(n_vocab_char):
  # main
    char_input = Input(shape=(None,), name='main_input')
    x = Embedding(input_dim=n_vocab_char,
                  output_dim=embed_size,
                  #mask_zero=True,
                  trainable=True)(char_input)
    x = Bidirectional(LSTM(n_lstm, return_sequences=True))(x)
    crf=CRF(n_entity, sparse_target=True)
    output = crf(x)

    model = Model(inputs=[char_input],
                       outputs=output)
    model.compile(optimizer="adam",
                       loss=crf.loss_function,
                       metrics=[crf.accuracy])
    model.summary()
    return model

def train():


  from sklearn.model_selection import train_test_split

  df_new = read_data_from_pickle(agg=True)
  texts=df_new['texts'].map(lambda x:list(x)).values
  texts_id ,n_vocab_char= word2id(texts)

  labels = df_new['labels'].values
  labels_id=pad_sequences(labels,maxlen=MAXLEN,truncating='post',padding='post',value=0)

  print('labels_id shape',labels_id.shape)
  print('texts_id',texts_id.shape)

  x_train,x_dev,y_train,y_dev=train_test_split(texts_id[:-1000],labels_id[:-1000] ,test_size=0.2, random_state=42)
  
  print('y_train',y_train.shape)

  y_train=np.expand_dims(y_train, 2) 
  y_dev=np.expand_dims(y_dev, 2) 

  model = get_model(n_vocab_char)
  K.set_value(model.optimizer.lr, 0.01)
  model.fit(x_train,y_train,epochs=10,batch_size=256,validation_data=(x_dev,y_dev))
  predict(model,texts_id,df_new)

def get_df_val():
  df_new = read_data()
  df_val = df_new[20000:]
  df_val.reset_index(inplace=True)
  return df_val


def decoder(text_ids,texts):
  final_res =[]
  for i in range(len(text_ids)):
      text = texts[i]
      text_id =text_ids[i]
      res = ''
      for j in range(len(text_id)):
          if(text_id[j]==1 or text_id[j]==0):
              pass
          else:
              if(text_id[j])==2:
                  res =res+' '+text[j]
              else:
                  res=res+text[j]
      final_res.append(res)
  return final_res


from collections import Counter
def post(x):
    #格式处理：c
    x = np.array(x).flatten().tolist()
    x=' '.join(x)
    x = x.split() 
    #过滤

    return[i[0] for i in  Counter(x).most_common(3)]



def postposs(df):
  '''

  聚合结果
  '''


  df.fillna(' ',inplace=False)
  feat_agg = ['predict','entity',]
  gg = df.groupby(['newsId'])
  df_agg = gg[feat_agg[0]].apply(lambda x:' '.join(x)).reset_index()
  for f in feat_agg:
      df_agg[f] = gg[f].apply(list).reset_index()[f]
  df_agg['entity']= df_agg['entity'].map(lambda x:x[0])

  
  return df_agg 



def predict(model,texts_id,df_new):

  df_val = df_new[-100:]
  df_val.reset_index(inplace=True)

  texts=df_val['texts'].values

  y = np.argmax(model.predict(texts_id[-100:]),axis=2)


  final_res = decoder(y,texts)
  df_val['predict']= final_res

  #df_val=postposs(df_val)
  df_val['predict'] = df_val['predict'].map(lambda x:post(x))

  #df_val['texts'] = df_val['texts'].map(lambda x:' '.join(x))
  
  df_val[['entity','newsId','predict']].to_csv('res.csv', sep='\t',index=False)




def main():
  train()
  
if __name__ == '__main__':
  main()

