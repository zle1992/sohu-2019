import pandas as pd
import re
import numpy as np
from collections import Counter



starDict= pd.read_csv('./dict/person.txt',names=['entity']).entity.to_list()
starDict = {k:v for v,k in enumerate(set(starDict))}


nerlist = pd.read_csv('../data/nerDict.txt',names=['entity']).entity.to_list() \
 + pd.read_csv('./dict/person.txt',names=['entity']).entity.to_list()

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
    badlist = ['的','是','\\','：',"》"]
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





def read_df(path,flag,sep=' '):
    df = pd.read_csv(path,sep='\t',names=['newsId','entity','entity_all','emtion'])
    df = df.fillna('')

    df = df.astype(str)
    df['entity_all'] = df['entity_all'].map(lambda x :x.replace('\'','').replace('[','').replace(']',''))
    # if flag =='content_title':
    #     df['entity_all'] = df['entity_all'].map(lambda x:x[2:-2])
    
    
    #clean
    df['entity_all']=df['entity_all'].map(lambda x: x.split(sep))

    #过滤长度为1的字符
    df['entity_sub']=df['entity_all'].map(lambda x:[i for i in x if x!=' '])
    
    df['entity_sub']=df['entity_sub'].map(lambda x:u200b(x))
    df['entity_sub']=df['entity_sub'].map(lambda x:book(x))
    
    #df['entity_sub']=df['entity_sub'].map(lambda x:huaweip30(x))
    #df['entity_sub']=df['entity_sub'].map(lambda x:huaweipnova4e(x))

    #df['entity_sub']=df['entity_sub'].map(lambda x:star_name_filter(x))
    #
   

    return df


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
    x=' '.join(x)
    x = x.replace(',', '')
    x = x.split() 
    d =[]
    line = Counter(x).most_common(3)
    #line=[(l,k) for k,l in sorted([(j,i) for i,j in Counter(x).items()], reverse=True)]
    line_cnt = float(np.array([i[1] for i in line]).sum())
    for entity_cnt in  line:
        #0.30 :
        #0.25 :   0.6469      entity_13794
        #0.20           0.648  entity_141381.txt
        #0.15 : 12+13: 0.591 145643.txt
        if len(line)>1 and entity_cnt[1]/(line_cnt+0.0001)<0.20 and entity_cnt[0] not in nerDict:
            pass
        else:
            d.append(entity_cnt[0])
    return d[:3]

# def post4rank(x):
#      #格式处理：c
#     x = np.array(x).flatten().tolist()
#     x=','.join(x)
#     #x = x.replace(',', '')
#     x = x.split(',') 
#     d =[]
#     line = Counter(x).most_common()
#     line_cnt = float(np.array([i[1] for i in line]).sum())
#     for entity_cnt in  line:
#         if len(line)>1 and entity_cnt[1]/(line_cnt+0.0001)<0.20 and entity_cnt[0] not in nerDict:
#             pass
#         else:
#             d.append(entity_cnt[0])
#     return d







def run(paths1,paths2,flag):
    df_sub=pd.DataFrame()
    
    for i,path in enumerate(paths1):
        print(path)
        df3 = read_df(path,flag,sep=' ')
        print(df3.head())
        if i==0:
            df_sub['newsId'] =df3['newsId']
            df_sub['entity_all'] = df3['entity_sub']
        else:
            df_sub['entity_all'] +=df3['entity_sub']
            df_sub['newsId'] =df3['newsId']
    if paths2 :
        for i,path in enumerate(paths2):
            print(path)

            df3 = read_df(path,flag,sep=',')
            print(df3.head())
            if len(paths1)==0 and i==0:
                df_sub['newsId'] =df3['newsId']
                df_sub['entity_all'] = df3['entity_sub']
            else:
                df_sub['entity_all'] +=df3['entity_sub']
                df_sub['newsId'] =df3['newsId']

    rules(df_sub)


def rules(df_sub):         
    print(df_sub.head())
    #过滤长度为1的字符
    df_sub['entity_all']=df_sub['entity_all'].map(lambda x:[i for i in x if len(i)>1])

    df_sub['entity_all']=df_sub['entity_all'].map(lambda x:u200b(x))
    df_sub['entity_all']=df_sub['entity_all'].map(lambda x:book(x))



    # df_sub['entity_rank']=df_sub['entity_all'].map(lambda x:post4rank(x))
    # df_sub['entity_rank']=df_sub['entity_rank'].map(lambda x:star_name_filter(x))
    # df_sub['entity_rank']=df_sub['entity_rank'].map(lambda x:vivo2x7(x))
    # df_sub['entity_rank']=df_sub['entity_rank'].map(lambda x:delepro(x))


    df_sub['entity_sub']=df_sub['entity_all'].map(lambda x:post(x))
    df_sub['entity_sub']=df_sub['entity_sub'].map(lambda x:star_name_filter(x))
    df_sub['entity_sub']=df_sub['entity_sub'].map(lambda x:vivo2x7(x))
    df_sub['entity_sub']=df_sub['entity_sub'].map(lambda x:delepro(x))


    df_sub['emotion_pred']=df_sub.apply(lambda x:['POS']*len(x['entity_sub']),axis=1)
    df_sub['entity_pred'] = df_sub['entity_sub'].map(lambda x:','.join(x))
    df_sub['emotion_pred'] = df_sub['emotion_pred'].map(lambda x:','.join(x))

    kk = df_sub.entity_pred.map(lambda x:len(x.split(','))).sum()
    print(kk)
    print(df_sub.head())
    df_sub[['newsId','entity_pred','emotion_pred']].to_csv('./output/entity_%d.txt'%kk, sep='\t',index=False,header=False)


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
    path18='../BERT-BiLSTM-CRF-NER/second_model/content_title/output_500_no_agg_234_lstm_128/bert_res.txt.all_pred.csv'
    paths1 =[
    #path3,
    #path4,
    #path5,
    #path6,
    #path7,
    #path10,
    path12,
    path13,
    
    ]



    #346  entity_160361  0.61163816
    #6 12  entity_144337  0.63094
    #12 13 0.645


    paths2 =[

    #path14,
    #path16,
    path18,
    ]
    run(paths1,paths2,'content_title')




def title_50():
    #0.5762
    path1='../BERT-BiLSTM-CRF-NER/title_50_234_lstm_128/bert_res.txt.all_pred.csv'
    #0.5649
    path2='../BERT-BiLSTM-CRF-NER/title_50_233_lstm_128/bert_res.txt.all_pred.csv'
    #0.5804
    path3='../BERT-BiLSTM-CRF-NER/title_50_234_lstm_256/bert_res.txt.all_pred.csv'
    #0.5939
    path4='../BERT-BiLSTM-CRF-NER/title_50_233_lstm_256/bert_res.txt.all_pred.csv'

    

    paths1 =[
    #path1,
    #path2,
    #path3,
    #path4,
    
    ]
    
    #0.574 5eoch
    path1='../BERT-BiLSTM-CRF-NER/second_model/title/50_no_agg_233/bert_res.txt.all_pred.csv'
    
    #0.544
    path2='../BERT-BiLSTM-CRF-NER/second_model/title/50_no_agg_233_new/bert_res.txt.all_pred.csv'
    #0.566
    path3='../BERT-BiLSTM-CRF-NER/second_model/title/50_no_agg_233_new2/bert_res.txt.all_pred.csv'

    paths2 = [
    path3,
    ]
    run(paths1,paths2,'title')
    
def main():
    #pall_500()
    title_50()
    #run(paths,'content_title')

if __name__ == '__main__':
    main()
