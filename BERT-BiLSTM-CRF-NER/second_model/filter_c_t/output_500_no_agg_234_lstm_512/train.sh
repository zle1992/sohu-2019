#!/bin/sh
model_path="/home/gpu401/lab/bigdata/zhangle/2019/zhangle11/bert/chinese_L-12_H-768_A-12"


type="234"


Train_path="/home/gpu401/lab/bigdata/sohu-2019/data2/coreEntityEmotion_train.txt.filter.pick"
Test_path="/home/gpu401/lab/bigdata/sohu-2019/data2/coreEntityEmotion_test_stage1.txt.filter.pick"


lstm_dim=512
Out_dir="/home/gpu401/lab/bigdata/sohu-2019/BERT-BiLSTM-CRF-NER/second_model/filter_c_t/output_500_no_agg_234_lstm_512"

echo ${Out_dir}

nohup python ../../../bert_base/train/bert_lstm_ner.py \
    --train_path=${Train_path} \
    --test_path=${Test_path} \
    --output_dir=${Out_dir} \
    --vocab_file=${model_path}/vocab.txt \
    --bert_config_file=${model_path}/bert_config.json \
    --init_checkpoint=${model_path}/bert_model.ckpt  \
    --max_seq_length=500 \
    --crf_only=False\
    --title_only=False \
    --batch_size=4 \
    --learning_rate=2e-5 \
    --do_train=True\
    --do_eval=True \
    --do_predict=True \
    --testisdev=False \
    --trainisdev=False  \
    --num_train_epochs=5 \
    --lstm_size=${lstm_dim} \
    --label_type=${type} \
    --device_map=0 >>${Out_dir}/train.log &
