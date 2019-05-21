import pandas as pd
import re
import numpy as np
from collections import Counter

import sys

sys.path.append('./common')
from post_rule import rule4all,rule4sub,df_submit
from read import clean_entity

def read_df(path,flag,sep=' '):
    df = pd.read_csv(path,sep='\t',names=['newsId','entity','entity_all','emtion'])
    df = df.fillna('')

    df = df.astype(str)
    df['entity_all'] = df['entity_all'].map(lambda x :x.replace('\'','').replace('[','').replace(']',''))
    df['entity_all']=df['entity_all'].map(lambda x: x.split(sep))
    df['entity_all']= df['entity_all'].map(lambda x:[clean_entity(i) for i in x])

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
    root_path = 'BERT-BiLSTM-CRF-NER/fist_state_model/content_title/'

    #no_agg_234_256_134810   0.631
    path12=root_path+'output_500_no_agg_234_lstm_256/bert_res.txt.all_pred.csv'
        
    #no_agg_234_lstm_512_138740  0.636
    path13 = root_path+'output_500_no_agg_234_lstm_512/bert_res.txt.all_pred.csv'


    root_path = 'BERT-BiLSTM-CRF-NER/second_model/content_title/'


    path17=  root_path+'output_500_no_agg_233/bert_res.txt.all_pred.csv'
    path18= root_path+'output_500_no_agg_234_3epoch/bert_res.txt.all_pred.csv'
    path19= root_path+'output_500_no_agg_234/bert_res.txt.all_pred.csv'
    path20=root_path+'output_500_no_agg_233_3epoch/bert_res.txt.all_pred.csv'
    root_path = 'BERT-BiLSTM-CRF-NER/second_model/filter_c_t/'
    path21=root_path+'output_500_no_agg_234/bert_res.txt.all_pred.csv'
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

    # path17,
    # path18,
    # path19,
    path21,
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
    
    
from six.moves import urllib

root_path = './'
special_dic=['\u2002','\u2003','\u3000','\u2028']

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    for tmp in special_dic:
        s = s.replace(tmp,' ')
    def _func(matched):
        return urllib.parse.unquote(matched.group(0))    
    patt = re.compile('(%[0-9a-fA-F]{2})+')
    s = re.sub(patt, _func, s)
    return(s)  

 
 #' 半角转全角 
def SBC2DBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            inside_code += 0xfee0
        rstring += chr(inside_code)
    return rstring
# 全角转半角
def q_to_b(q_str):
    b_str = ""
    for uchar in q_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        b_str += chr(inside_code)
    return b_str
# 清洗字符串
httpcom = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配连接
space = re.compile(r' +') # 将一个以上的空格替换成一个空格
link = re.compile(r'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+') # 匹配网址
repeat = re.compile(r'(.)\1{5,}') # 超过6个以上的连续字符匹配掉比如......，人人人人人人
mm = re.compile("[()（）\\n *  %《*》•、&＆(—)（+）：“”【】]+")


def clean_text(raw):
    raw = q_to_b(raw)
    raw = mm.sub('', raw)
    raw = httpcom.sub('', raw)
    raw = space.sub(' ', raw)
    raw = link.sub('', raw)
    raw = repeat.sub('', raw)
    raw = raw.replace('...', '。').replace('！ ！ ！', '！').replace('！ 。', '！').replace('？ 。', '？')
    raw=replace_html(raw)
    raw = raw.replace('[', '').replace(']','')
    raw = raw.replace(',', '，')
    return raw
def clean_entity(entity):
    entity =entity.strip()
    entity = entity.lower()
    entity=clean_text(entity)
    return entity
    
def run(paths1,paths2,flag):
  
    df_sub =merge_df(paths1,paths2,flag)

    df_sub['entity_all'] = df_sub['entity_all'].map(lambda x:[clean_entity(u) for u in x])
    out_dir ='model_combine/output/combine_'
    df_sub ,entity_num=df_submit(df_sub)
    df_sub[['newsId','entity_sub','emotion_pred']].to_csv(out_dir+'entity_%d.txt'%entity_num, sep='\t',index=False,header=False)
def main():
    all_500()
    #title_50()
    #run(paths,'content_title')
if __name__ == '__main__':
    main()