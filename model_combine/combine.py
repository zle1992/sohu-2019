import pandas as pd
import re
import numpy as np
from collections import Counter

import sys

sys.path.append('./common')
from post_rule import rule4all,rule4sub,df_submit


def read_df(path,flag,sep=' '):
    df = pd.read_csv(path,sep='\t',names=['newsId','entity','entity_all','emtion'])
    df = df.fillna('')

    df = df.astype(str)
    df['entity_all'] = df['entity_all'].map(lambda x :x.replace('\'','').replace('[','').replace(']',''))
    df['entity_all']=df['entity_all'].map(lambda x: x.split(sep))

    df=rule4all(df)  

    return df


def merge_df(paths1,paths2,flag):
    df_sub=pd.DataFrame()
    
    for i,path in enumerate(paths1):
        print(path)
        df3 = read_df(path,flag,sep=' ')
        print(df3.head())
        if i==0:
            df_sub['newsId'] =df3['newsId']
            df_sub['entity_all'] = df3['entity_all']
        else:
            df_sub['entity_all'] +=df3['entity_all']
            df_sub['newsId'] =df3['newsId']
    if paths2 :
        for i,path in enumerate(paths2):
            print(path)

            df3 = read_df(path,flag,sep=',')
            print(df3.head())
            if len(paths1)==0 and i==0:
                df_sub['newsId'] =df3['newsId']
                df_sub['entity_all'] = df3['entity_all']
            else:
                df_sub['entity_all'] +=df3['entity_all']
                df_sub['newsId'] =df3['newsId']

    return df_sub



def all_500():
    root_path = '../BERT-BiLSTM-CRF-NER/fist_state_model/content_title/'
    
    #no_agg_233_165628.txt   0.601
    path3=root_path+'output_500_no_agg_233/bert_res.txt.all_pred.csv'
    #no_agg_234_138744     0.609
    path4=root_path+'output_500_no_agg_234/bert_res.txt.all_pred.csv'
    #only_ctf
    path5=root_path+'output_500_agg_234/bert_res.txt.all_pred.csv'

    #    no_agg_234_lstm_128_151728 0.6161
    path6=root_path+'output_500_no_agg_234_lstm_5epoch/bert_res.txt.all_pred.csv'

    #lstm 128
    path7=root_path+'output_500_agg_234_lstm/bert_res.txt.all_pred.csv'
    #lstm 128
    path10=root_path+'output_500_agg_233_lstm/bert_res.txt.all_pred.csv'


    #no_agg_234_256_134810   0.631
    path12=root_path+'output_500_no_agg_234_lstm_256/bert_res.txt.all_pred.csv'
        
    #no_agg_234_lstm_512_138740  0.636
    path13 = root_path+'output_500_no_agg_234_lstm_512/bert_res.txt.all_pred.csv'


    root_path = '../BERT-BiLSTM-CRF-NER/fist_state_model/new_agg/'


    path14 = root_path+'output_500_agg_234/bert_res.txt.all_pred.csv'

    path15 = root_path+'output_500_agg_234_lstm_56/bert_res.txt.all_pred.csv'

    root_path = '../BERT-BiLSTM-CRF-NER/second_model/agg/'
    

    path16 = root_path+'output_500_agg_234_lstm_256/bert_res.txt.all_pred.csv'

   # path17 = root_path+'output_500_agg_234_lstm_56/bert_res.txt.all_pred.csv'
   # 0.49
    #'../BERT-BiLSTM-CRF-NER/fist_state_model/new_agg/output_500_agg_234_lstm_56/bert_res.txt.all_pred.csv'
    path18='./BERT-BiLSTM-CRF-NER/second_model/content_title/output_500_no_agg_234_3epoch/bert_res.txt.all_pred.csv'
    path19='./BERT-BiLSTM-CRF-NER/second_model/content_title/output_500_no_agg_234/bert_res.txt.all_pred.csv'
    path20='./BERT-BiLSTM-CRF-NER/second_model/content_title/output_500_no_agg_233_3epoch/bert_res.txt.all_pred.csv'
    paths1 =[
    #path3,
    #path4,
    #path5,
    #path6,
    #path7,
    #path10,
    #path12,
    #path13,
    
    ]



    #346  entity_160361  0.61163816
    #6 12  entity_144337  0.63094
    #12 13 0.645


    paths2 =[

    #path14,
    #path18,
    #path19,
    path20,
    ]
    run(paths1,paths2,'content_title')




def title_50():
    #0.5762
    path1='./BERT-BiLSTM-CRF-NER/title_50_234_lstm_128/bert_res.txt.all_pred.csv'
    #0.5649
    path2='./BERT-BiLSTM-CRF-NER/title_50_233_lstm_128/bert_res.txt.all_pred.csv'
    #0.5804
    path3='./BERT-BiLSTM-CRF-NER/title_50_234_lstm_256/bert_res.txt.all_pred.csv'
    #0.5939
    path4='./BERT-BiLSTM-CRF-NER/title_50_233_lstm_256/bert_res.txt.all_pred.csv'

    paths1 =[
    #path1,
    #path2,
    #path3,
    #path4,
    
    ]
    
    #0.574 5eoch
    path1='./BERT-BiLSTM-CRF-NER/second_model/title/50_no_agg_233/bert_res.txt.all_pred.csv'
    #0.544
    path2='../BERT-BiLSTM-CRF-NER/second_model/title/50_no_agg_233_new/bert_res.txt.all_pred.csv'
    #0.566
    #path3='../BERT-BiLSTM-CRF-NER/second_model/title/50_no_agg_233_new2/bert_res.txt.all_pred.csv'

    paths2 = [
    path2,
    ]
    
    
    run(paths1,paths2,'title')
    
    
def run(paths1,paths2,flag):
  
    df_sub =merge_df(paths1,paths2,flag)
    out_dir ='model_combine/output/combine_'
    df_sub ,entity_num=df_submit(df_sub)
    df_sub[['newsId','entity_sub','emotion_pred']].to_csv(out_dir+'entity_%d.txt'%entity_num, sep='\t',index=False,header=False)
def main():
    all_500()
    #title_50()
    #run(paths,'content_title')

if __name__ == '__main__':
    main()
