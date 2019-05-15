import pandas as pd
import json
import keras
import numpy as np
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import re
import numpy as np


def text2id(x,col='content',flag='233',rep=' '):
    content =x[col]
    content_label =len(content) *[1]
    content_label = np.array(content_label)
    try:
       
        pattern =x['entity'].replace(rep,'|')
        it = re.finditer(pattern,content) 
        for match in it: 
          start,end = match.span()
          if(start==end):
              continue
          content_label[start]=2 
          if flag=='233':
              content_label[(start+1):end]=3
          
          else:
            if end-start>2:

              content_label[(start+1):end-1]=3
            content_label[end-1]=4
            
    except:
        pass
    return list(content_label)



# def decoder(text_ids,texts,rep=' '):
#   final_res =[]
#   for i in range(len(texts)):
#       text = texts[i]
#       text_id =text_ids[i]
#       res = ''
#       ll = min(len(text_id),len(text))
#       for j in range(ll):
#           if(text_id[j]==1 or text_id[j]==0):
#               pass
#           else:
#               if(text_id[j])==2:
#                   res =res+rep+text[j]
#               else:
#                   res=res+text[j]
#       final_res.append(res[1:])
#   return final_res





def result_to_json(string, tags, ):
    item = {"string": string, "entities": []}
    entity_name = ""
    idx = 0

    for i, (char, tag) in enumerate(zip(string, tags)):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            for j in range(i + 1, len(tags)):
                entity_name += string[j]
                if tags[j][0] == 'I':
                    continue
                elif tags[j][0] == 'E':
                    item["entities"].append({"word": entity_name, "start": idx, "end": j + 1, "type": tag[2:]})
                else:
                    break
            entity_name = ""
        idx += 1
    return item

LAB_MAP = { "B": 2,"I": 3,"E": 4,"O":1}
id2label = {0:'O',1:'O',2:'B',3:'I',4:'E'}


def decoder(text_ids,texts,flag='234'):
  '''
  string=['v','i','v','o','手','机']
  tags=['B-POS','I-POS','I-POS','E-POS','O','O']
  '''

  final_res =[]
  for i in range(len(texts)):
      text = texts[i]
      text_id =text_ids[i]
      text_id = [id2label[i] for i in text_id]

      res =[]
      for info in result_to_json(text,text_id)['entities']:
        res.append((info['word']))
      final_res.append(','.join(res))
  return final_res



# def text2id(x,col='content'):
#     content =x[col]
#     content_label =len(content) *[1]
#     content_label = np.array(content_label)
#     try:
       
#         pattern =x['entity'].replace(' ','|')
#         it = re.finditer(pattern,content) 
#         for match in it: 
#           start,end = match.span()
#           if(start==end):
#               continue
#           content_label[start]=2 
#           content_label[(start+1):end]=3
#     except:
#         pass
#     return list(content_label)



# def decoder(text_ids,texts):
#   final_res =[]
#   for i in range(len(texts)):
#       text = texts[i]
#       text_id =text_ids[i]
#       res = ''
#       ll = min(len(text_id),len(text))
#       for j in range(ll):
#           if(text_id[j]==1 or text_id[j]==0):
#               pass
#           else:
#               if(text_id[j])==2:
#                   res =res+' '+text[j]
#               else:
#                   res=res+text[j]
#       final_res.append(res)
#   return final_res


from collections import Counter
def post(x):
    #格式处理：c
    x = np.array(x).flatten().tolist()
    x=','.join(x)
    x = x.split(',') 
    #过滤

    return[i[0] for i in  Counter(x).most_common(3)]

def postposs(df,):
  '''

  聚合结果
  '''

  df.fillna(' ',inplace=False)
  print(list(df))
  feat_agg = ['entity_pred','entity',]
  gg = df.groupby(['newsId'])
  df_agg = gg[feat_agg[0]].apply(lambda x:','.join(x)).reset_index()
  for f in feat_agg:
      df_agg[f] = gg[f].apply(list).reset_index()[f]
  df_agg['entity']= df_agg['entity'].map(lambda x:x[0])
  print(list(df_agg))

  return df_agg 




import numpy as np
def score(reals,preds):
    '''
    real=['a','b','c']
    pred=['a','b']
    
    score([['a'],['a','b']],[['b'],['a']])
    '''
    pl,rl,f1l=[],[],[]
    for real_s,pred_s in zip(reals,preds):
        real = real_s.split(',')
        pred = pred_s.split(',')
        tp =float(len(set(pred)&set(real)))
        p=tp/len(pred)+0.000001
        r=tp/len(real)+0.000001
        f1=2*(p*r)/(p+r)
        pl.append(p)
        rl.append(r)
        f1l.append(f1)
    p = np.array(pl).mean()
    r = np.array(rl).mean()
    f1 = np.array(f1l).mean()
    
    print('precision: ',p)
    print('recall: ',r)
    print('f1: ',f1)
    return p,r,f1



