#!/bin/sh
BERT_BASE_DIR="/home/gpu401/lab/bigdata/zhangle/2019/zhangle11/bert/chinese_L-12_H-768_A-12"
GLUE_DIR="./GLUE_DIR"
root_dir='/home/gpu401/lab/bigdata/sohu-2019'
Train_path=${root_dir}/data2/coreEntityEmotion_train.txt.df_emo2.csv
Test_path=${root_dir}/data2/coreEntityEmotion_test_stage1.txt.df_emo2.csv

out_dir=${root_dir}/bert_emo/df_emo2/

gpu_device='0'
batch_size=100
max_len=450



nohup python ./bert/run_classifier.py \
  --task_name=emotion \
  --train_path=$Train_path \
  --test_path=$Test_path \
  --gpu_device=$gpu_device \
  --do_train=False \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$root_dir \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=$max_len \
  --train_batch_size=$batch_size \
  --eval_batch_size=$batch_size \
  --predict_batch_size=$batch_size \
  --learning_rate=1e-6 \
  --num_train_epochs=3.0 \
  --output_dir=$out_dir >>${out_dir}log.log &
