sohu-bert

#实体
├── BERT-BiLSTM-CRF-NER   
│   ├── bert_base
│   │   ├── bert
│   │   └── train
│   │       ├── bert_lstm_ner.py   #bert_lstm_ner模型      
│   │       ├── conlleval.pl     
│   │       ├── conlleval.py    #评测文件     
│   │       ├── lstm_crf_layer.py   #Lstm_crf_layer       
│   │       ├── make_data.ipynb    
│   │       ├── models.py   #公共模块     
│   │       └── tf_metrics.py        
│   ├── second_model      
│   │   ├──title    
│   │   ├──content_title   
│   │      	   ├── output_500_agg_234 #模型结果产出   
│   │		      ├── bert_res.txt    #用于模型融合的结果   
│   │		      ├── bert_res.txt.all_pred.csv   #同上   
│   │		      ├── bert_res.txt.all_pred_epoch2.csv    #同上   
│   │		      ├── bert_res_epoch2.txt #一些中间文件   
│   │		      ├── checkpoint     
│   │		      ├── eval       
│   │		      ├── train.log   #   
│   │			  ├─ train.sh #训练脚本			     
│  
├── common  #一些共同文件    
│   ├── entity_rec.py   #词典匹配   
│   ├── esmre.py    
│   ├── post_rule.py #后处理规则    
│   ├── read.py #读取json    
│   └── util.py #辅助函数    
├── data    #训练集测试集存放的位置   
├── data2    #复赛数据    
├── model_combine   #模型融合，及最后融合结果的产出   
│   ├── combine.py  #融合   
│   ├── dict    #开源词典   
│   └── output  #结果文件    
│       └── entity_90209.txt   
#情感   
├── bert_emo   
│   ├── bert    
│   │   ├── bert
│   │   ├── make_data.py #生成句子对文件   
│   │ 	└──	df_emo2   
│   │       ├── train.sh  #训练脚本   
│   │       ├── test_results.csv #预测结果     

