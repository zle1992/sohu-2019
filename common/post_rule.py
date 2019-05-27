import pandas as pd
import re
import numpy as np
from collections import Counter

root_dir = '/home/gpu401/lab/bigdata/sohu-2019/'

starDict= pd.read_csv(root_dir+'data/dict/person.txt',names=['entity']).entity.to_list()
starDict = {k:v for v,k in enumerate(set(starDict))}


nerlist = pd.read_csv(root_dir+'data/nerDict.txt',names=['entity']).entity.to_list() \
 + pd.read_csv(root_dir+'data/dict/person.txt',names=['entity']).entity.to_list()

nerset = nerlist
nerDict = {k:v for v,k in enumerate(nerset)}


def u200b(x):
    '''
    去除 u200b
    '''
    res = []
    for s in x:
        if not re.search('\u200b',s) and not re.search('u200b',s):
            res.append(s)
    return res

def book(x):
    '''
    去除第一个char is :
    '的','是','\\','：',"》"
    '''
    badlist = ['的','是','\\','：',"》",'，','。']
    fist_baddict ={k:v for v,k in enumerate(badlist)}
    res = []
    for s in x:
        if len(s)<1:
            continue
        if re.search('《',s) or s[0] in fist_baddict or s[-1] in ["："] :
            pass
        else:
            res.append(s)
    return res


#*******************************************************************
def huaweip30(x):
    '''
    '华为p30' -->'华为','p30'
    '''
    res = []
    for entity in x:
        if entity=='华为p30':
            res.append('华为')
            res.append('p30')
        else:
            res.append(entity)
    return res

def huaweipnova4e(x):
    '''
    '华为nova4e' ->'华为','nova4e'
    '''
    res = []
    for entity in x:
        if entity=='华为nova4e':
            res.append('华为')
            res.append('nova4e')
        else:
            res.append(entity)
    return res

def star_name_filter(ss):
    '''
        ss = ['王思聪','微信','王思']
    
    out:['王思聪', '微信']
    '''
    final_res=[]
    name_reals =[s for s in ss if s in starDict]
    #print(name_reals)
    bad_name = []
    for s in ss:
        if s in name_reals:
            
            final_res.append(s)
            continue
        else:
            for name_real in name_reals:
                if len(name_real) >3:
                    continue
                pattern = '?.'+name_real[:2]+'|'+name_real[:2]+'.?'
                pattern = '[\u4e00-\u9fa5]?'+name_real[:2]+'$|'+name_real[:2]+'[\u4e00-\u9fa5]?$' # match 第一个参数是需要匹配的字符串，第二个是源字符串
                if re.match(pattern,s): # match 第一个参数是需要匹配的字符串，第二个是源字符串
                    bad_name.append(s)
                    pass
                
                    
    final_res = [s for s in ss if s not in bad_name]
    return final_res

def vivo2x7(x):
    #将所有的vivox27换为vivo x27，只有vivox27 x27 vivo三个的将后2个删除
    
    if set(x)==set(['vivo','x27','vivox27']):
        return ['vivo x27']
    res = []
    for entity in x:
        if entity=='vivox27':
           res.append('vivo x27')
        else:
            res.append(entity)
    
    ff =[]
    for entity in res:
        if entity =='x27':
            ff.append('vivo x27')
        else:
            ff.append(entity)
    ff = list(set(ff))
    return ff


def delepro(x):
    res =[]
    for entity in x:
        if entity =='pro':
            pass
        else:
            res.append(entity)
    return res



def post(x):
     #格式处理：c
    x = np.array(x).flatten().tolist()
    #x=','.join(x)
    #x = x.replace(',', '')
    #x = x.split() 
    d =[]
    line = Counter(x).most_common(3)
    #line=[(l,k) for k,l in sorted([(j,i) for i,j in Counter(x).items()], reverse=True)]
    line_cnt = float(np.array([i[1] for i in line]).sum())
    for entity_cnt in line:
        #0.30 :
        #0.25 :   0.6469      entity_13794
        #0.20           0.648  entity_141381.txt
        #0.15 : 12+13: 0.591 145643.txt
        if len(line)>1 and entity_cnt[1]/(line_cnt+0.0001)<0.20 and entity_cnt[0] not in nerDict:
            pass
        else:
            d.append(entity_cnt[0])
    return d[:3]


def rule4all(df):
    
    #过滤长度为1的字符
    df['entity_all']=df['entity_all'].map(lambda x:[i for i in x if len(i)>1])
    df['entity_all']=df['entity_all'].map(lambda x:[i for i in x if x!=' '])
    df['entity_all']=df['entity_all'].map(lambda x:u200b(x))
    df['entity_all']=df['entity_all'].map(lambda x:book(x))
   
    return df

def rule4sub(df_sub):
 
    df_sub['entity_sub']=df_sub['entity_sub'].map(lambda x:star_name_filter(x))
    df_sub['entity_sub']=df_sub['entity_sub'].map(lambda x:vivo2x7(x))
    df_sub['entity_sub']=df_sub['entity_sub'].map(lambda x:delepro(x))
    
    return df_sub

def rule4combine(df_sub):
    
    df_sub=rule4all(df_sub)
    df_sub['entity_sub']=df_sub['entity_all'].map(lambda x:post(x))
    df_sub = rule4sub(df_sub)
    return df_sub


def rule4bert(df_sub):

    df_sub=rule4all(df_sub) 
    df_sub['entity_sub']=df_sub['entity_all'].map(lambda x:[i[0] for i in  Counter(x).most_common(3)])
    df_sub = rule4sub(df_sub)
    return df_sub

def df_submit(df_sub,flag='combine'):
    if flag=='bert':
        df_sub = rule4bert(df_sub)
    if flag=='combine':
        df_sub = rule4combine(df_sub)

   
    df_sub['emotion_pred']=df_sub.apply(lambda x:['POS']*len(x['entity_sub']),axis=1)
    df_sub['entity_sub'] = df_sub['entity_sub'].map(lambda x:','.join(x))
    df_sub['emotion_pred'] = df_sub['emotion_pred'].map(lambda x:','.join(x))
    df_sub['entity_all'] = df_sub['entity_all'].map(lambda x:','.join(x))

    print(df_sub['entity_sub'][0])
    print(df_sub.head())
    entity_num = df_sub.entity_sub.map(lambda x:len(x.split(','))).sum()
    print('entity num :',entity_num)
    return df_sub ,entity_num
   
