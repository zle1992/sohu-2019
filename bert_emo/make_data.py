import pandas as pd
import re


def get_df_emotion(df):
    newlines = []
    for i,line in df.iterrows():
        sentences = line['texts'].split('。')
        entitys = line['entity'].split(',')
        emotions = line['emotion'].split(',')
        for j ,entity in enumerate(entitys):
            sen_temp = []
            for sentence in sentences:
                try:
                    if entity[0] in ['*','+']:
                        entity='\\'+entity
                    if  entity=='c++':
                        entity='c\++'
                    if re.search(entity,sentence):
                        sen_temp.append(sentence)
                except:
                    pass

            newlines.append([line['newsId'],'。'.join(sen_temp),entity,emotions[j]])

    df_emotion = pd.DataFrame(newlines,columns=['newsId','sentence','entity','emotion'])    
    df_emotion.dropna(inplace=True)
    return df_emotion


def run_train():

    train_path='/home/gpu401/lab/bigdata/sohu-2019/data2/coreEntityEmotion_train.txt.pick'
    out_path = train_path.replace('pick','df_emo2.csv')


    df_train = pd.read_pickle(train_path)
    df_train['emotion']=df_train['emotion'].map(lambda x:x.replace(' ',''))
    df_train.emotion.map(lambda x:len(x.split(','))).sum()


    df_emotion = get_df_emotion(df_train)

    df_emotion[['newsId','sentence','entity','emotion']].to_csv(out_path,index=False,sep='\t')


def run_test(path_sub):
    test_path = '/home/gpu401/lab/bigdata/sohu-2019/data2/coreEntityEmotion_test_stage1.txt.pick'

    out_path = test_path.replace('pick','df_emo2.csv')

    
    df_sub = pd.read_csv(path_sub,sep='\t',names=['newsId','entity','emotion'])
    df_sub = df_sub.astype(str)
    df_test = pd.read_pickle(test_path)
    df_test = pd.merge(df_test.drop(['entity','emotion'],axis=1),df_sub,on='newsId',how='left')
    df_test['entity'] = df_test['entity'].map(lambda x:','.join(x.split(',')))
    df_test['emotion'] = df_test['emotion'].map(lambda x:','.join(x.split(',')))
    df_emotion = get_df_emotion(df_test)
    df_emotion[['newsId','sentence','entity','emotion']].to_csv(out_path,index=False,sep='\t')


def main():
    
    #run_train()
    run_test(path_sub = './bert_emo/entity_res/pred_0520_lgb_bert_merged (5).txt')
    
if __name__ == "__main__":
    main()
    
