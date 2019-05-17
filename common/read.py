#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
import numpy as np
import os
#import jieba
import sys
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
mm = re.compile(r"[()（）\\n *  %《*》•、&＆(—)（+）：“” 【】]+")


def clean_text(raw):
    raw = q_to_b(raw)
    raw = httpcom.sub('', raw)
    raw = space.sub(' ', raw)
    raw = link.sub('', raw)
    raw = repeat.sub('', raw)
    raw = raw.replace('...', '。').replace('！ ！ ！', '！').replace('！ 。', '！').replace('？ 。', '？')
    raw=replace_html(raw)
    raw = mm.sub('', raw)
    raw = raw.replace(',', '，')
    return raw

def clean_entity(entity):
    entity =entity.strip()
    entity = entity.lower()
    entity=clean_text(entity)
    return entity




def read_json(path,flag='train'):
    print(path)
    load_dicts =[]
    with open(path,'r') as f:
       
        for i ,jsondata in enumerate(f.readlines()):
            load_dict = json.loads(jsondata)
            if flag=='test':
                load_dict['coreEntityEmotions'] = [{'entity':' ','emotion':' '}]
            load_dicts.append(load_dict)

    df = pd.DataFrame(load_dicts)

    
    df['entity'] = df['coreEntityEmotions'].map(lambda x: ','.join([clean_entity(i['entity']) for i in x]))
    df['emotion'] = df['coreEntityEmotions'].map(lambda x: ','.join([i['emotion'] for i in x]))
  

    df['title'] = df['title'].map(lambda x:clean_text(x))
    df['content'] = df['content'].map(lambda x: clean_text(x))

    print('done')
    return df

# def jieba_cut(df):

#     e_dict = '../data/nerDict.txt'
#     #jieba.load_userdict(e_dict)

#     with open(e_dict,'w') as f:
#         f.writelines(' '.join(df['entity'].values).split())

    

#     df['title_cut']  =df['title'].map(lambda x:' '.join(jieba.cut(x)))

#     df['content_cut']  =df['content'].map(lambda x:' '.join(jieba.cut(x)))
#     return df




def cut_text(text,lenth=450): 
    res = []
    textArr=[]
    i =0
    if len(text)>lenth:
        t_list = re.split(r'[。！？]+', text)
        for tt in t_list:
            if len('。'.join(res))+len(tt)<lenth:
                res.append(tt)
            else:
                textArr.append('。'.join(res))
                res =[tt]
    else:
        t_list = re.split(r'[。！？]+', text)
        textArr = ['。'.join(t_list)]

    return textArr 


def make_new_df(df):

    contents,newsids,entitys ,titles,emotions,=[],[],[],[],[]

    for i in range(len(df)):
        for content in cut_text(df['content'][i],450):
            contents.append(content)
            newsids.append(df['newsId'][i])
            entitys.append(df['entity'][i])
            titles.append(df['title'][i])
            emotions.append(df['emotion'][i])
    df_new = pd.DataFrame([newsids,titles,contents,entitys,emotions]).T
    df_new.columns=['newsId','title','content','entity','emotion']
    return df_new




def run(flag):
    cut=False
    
    
    #if not os.path.exists(out_path):
    if flag=='train':
        out_path = root_path+'data2/coreEntityEmotion_train.txt'+'.pick'

        df = read_json(root_path+'data/coreEntityEmotion_train.txt',flag)
        df_sample =read_json(root_path+'data/coreEntityEmotion_example.txt',flag)
        df=df.append(df_sample, ignore_index=True)
    else:
        out_path = root_path+'data2/coreEntityEmotion_test_stage1.txt'+'.pick'
        df = read_json(root_path+'data/coreEntityEmotion_test_stage1.txt',flag)

    df['texts'] = df.apply(lambda x:x['title']+'。'+x['content'],axis=1)
    df.to_pickle(out_path,)
    

def get_agg_data(path):
    df = pd.read_pickle(path)
    print(df.shape)
    df = make_new_df(df)
    df['texts'] = df.apply(lambda x:x['title']+'。'+x['content'],axis=1)
    print(df.shape)
    out_path = path.replace('.pick','.agg.pick')
    df.to_pickle(out_path)
    print('done agg')



def get_filter_entity_data(path):

    def clean_entity(entity):
        if(len(entity))<1:
            return entity
        if entity[0] in ['*','+']:
                entity='\\'+entity
        if  entity=='c++':
                entity='c\++'
        return entity
    
    df = pd.read_pickle(path)
    newlines = []

    for i,line in df.iterrows():
        sentences = line['texts'].split('。')
        texts =[]
        for sentence in sentences:
            
            entitys = [clean_entity(i) for i in line['entity'].split(',')]
            partern = '|'.join(entitys)
           
            try:
                if re.search(partern,sentence):
                   texts.append(sentence)
            except: 
                #print(i,entitys,'!!!!!!!!!!!!!!!!!'+sentence)
                pass
            
        newlines.append([line['newsId'],'。'.join(texts),line['entity'],line['emotion']])
    
    df_new= pd.DataFrame(newlines,columns=['newsId','texts','entity','emotion'])
    df_new.dropna(inplace=True)

    out_path = path.replace('.pick','.filter.pick')
    print("filter out_path:",out_path)
    df.to_pickle(out_path)
    
def main():

    run(flag='train')
    run(flag='test')

    get_agg_data(root_path+'data2/coreEntityEmotion_train.txt.pick')
    get_agg_data(root_path+'data2/coreEntityEmotion_test_stage1.txt.pick')

    get_filter_entity_data(root_path+'data2/coreEntityEmotion_train.txt.pick')
    get_filter_entity_data(root_path+'data2/coreEntityEmotion_test_stage1.txt.pick')
if __name__ == '__main__':
    main()
