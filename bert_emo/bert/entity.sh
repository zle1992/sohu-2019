#!/bin/sh
BERT_BASE_DIR="/sdb1/zhangle/2019/zhangle11/bert/chinese_L-12_H-768_A-12"
GLUE_DIR="./GLUE_DIR"
#emotion_1_df
#train_path='/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_train.txt.pick.emotion_1_df.csv'
#test_path='/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_test_stage1.txt.pick.emotion_1_df.csv'
#out_dir='./mrpc_output/'
#batch_size=32
#max_len =50 


#entity_1_df
Train_path='/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_train.txt.esm_entity.csv.df_entity_1.csv'
Test_path='/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_test_stage1.txt.esm_entity.csv.df_entity_1.csv'
out_dir='./entity_1_df/'
gpu_device='0'
batch_size=4
max_len=500



python run_classifier.py \
  --task_name=entity \
  --train_path=$Train_path \
  --test_path=$Test_path \
  --gpu_device=$gpu_device \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=$max_len \
  --train_batch_size=$batch_size \
  --eval_batch_size=$batch_size \
  --predict_batch_size=$batch_size \
  --learning_rate=5e-6 \
  --num_train_epochs=1.0 \
  --output_dir=$out_dir
