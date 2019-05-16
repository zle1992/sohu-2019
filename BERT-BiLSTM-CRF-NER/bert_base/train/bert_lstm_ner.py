import collections
import os
import numpy as np
import tensorflow as tf
import codecs
import pickle




import sys
sys.path.append('../../../../')
sys.path.append('../../../../common')
from util import *


sys.path.append('../../../bert_base/bert')
import modeling, optimization,tokenization
# import
from models import create_model, InputFeatures, InputExample
import tf_metrics

# from bert_base.train import tf_metrics
# from bert_base.bert import modeling, optimization,tokenization

# # import

# from bert_base.train.models import create_model, InputFeatures, InputExample



import logging  
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # 

import tensorflow as tf  
import os  
#config = tf.ConfigProto()   
#config.gpu_options.allow_growth = True      #程序按需申请内存  
#sess = tf.Session(config = config)

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
## Required parameters
flags.DEFINE_string(
    "data_dir",  'NERdata',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
## Required parameters
flags.DEFINE_string(
    "bert_config_file", 'bert_config.json',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir",  'output',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool("testisdev", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("trainisdev", False, "Whether to run eval on the dev set.")


flags.DEFINE_integer(
    "max_seq_length", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "batch_size", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "dropout_rate", 0.5,"Total number")

flags.DEFINE_float(
    "clip", 0.5,"Total number")

flags.DEFINE_float(
    "warmup_proportion", 0.1,"Total number")

flags.DEFINE_integer(
    "lstm_size", 128,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_layers", 1,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_string(
    "cell", 'lstm',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 500,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "save_summary_steps", 500,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_bool("filter_adam_var", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_lower_case", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("clean", None, "Whether to run eval on the dev set.")

flags.DEFINE_bool("crf_only", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool("title_only", False, "Whether to run eval on the dev set.")
flags.DEFINE_string(
    "ner", "sohuner",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_string(
    "test_path", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_string(
    "train_path", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_integer(
    "device_map", 0,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "label_type", "233",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string(
    "rep", " ",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string(
    "root_path", "/home/gpu401/lab/bigdata/sohu-2019",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        pass

def find(x,sep=' '):
    sentence=x['title']
    entity='|'.join(x['entity'].split(sep))
    try:
        if re.search(entity,sentence):
            return True
    except:
        pass
    return False

class SOHUNERProcessor(DataProcessor):
  """Processor for the Emotion data set"""

  def title_filter(self,df):
    
    df = df[df.apply(lambda x:find(x,sep=FLAGS.sep),axis=1)]
      
    return df
  def df_process(self,df):
    df.fillna('null',inplace=True)
    df =df.astype(str)
    logging.info(list(df))
    if FLAGS.title_only :
      df['texts']=df['title']

    if FLAGS.rep==' ':    
        
        #转换成复赛的格式
        df['entity'] = df['entity'].map(lambda x:x.replace(' ',','))

    df['texts']=df['texts'].map(lambda x:x.replace(',','，'))
    
    #rep=',' 复赛连接符号。 只有text2id 才会用到rep!!!!!!
    df['labels']=df.apply(lambda x:text2id(x,col='texts',flag=FLAGS.label_type,rep=FLAGS.rep),axis=1)
    
    df['texts']=df['texts'].map(lambda x:list(x))
    df['texts'] = df['texts'].map(lambda x:','.join(x))
    df['labels'] = df['labels'].map(lambda x:','.join([str(i)for i in x]))

    print(df.head())
    # print(df.head()['texts'].map(lambda x:len(x.split(','))))
    # print(df.head()['labels'].map(lambda x:len(x.split(','))))

    return df


  def get_train_examples(self,):
    """See base class."""
    
    train_path = FLAGS.train_path
    tf.logging.info("train_path: %s" % (train_path))

    self.train_df = pd.read_pickle(train_path)

    k = int(len(self.train_df)*0.05)
    train_df = self.train_df[:-k].sample(frac =1,random_state=2019)

    train_df = self.df_process(train_df)
    #train_df=self.title_filter(train_df)

    return self._create_examples(train_df,"train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    k = int(len(self.train_df)*0.05)

    dev_df = self.train_df[-k:].sample(frac =1,random_state=2019)
    dev_df = self.df_process(dev_df)
    #dev_df=self.title_filter(dev_df)
    return self._create_examples(dev_df, "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    
    if FLAGS.testisdev or FLAGS.trainisdev :
        if FLAGS.testisdev:
            df_train=pd.read_pickle(FLAGS.train_path)[:]
            
            k = int(len(df_train)*0.05)
            self.test_df =  df_train[-k:]#df_train[-k:]
        if FLAGS.trainisdev:
            self.test_df = pd.read_pickle(FLAGS.train_path)
            #self.test_df=self.title_filter(self.test_df)
        
    else:
        test_path = FLAGS.test_path
        tf.logging.info("test_path: %s" % (test_path))
        self.test_df = pd.read_pickle(test_path)


    self.test_df = self.df_process(self.test_df)
    return self._create_examples(self.test_df, "test")

  def get_labels(self):
    """See base class."""
    if FLAGS.label_type=='233':
        self.labels = set(['0','1','2','3',"[CLS]", "[SEP]"])
    else:
        self.labels = set(['0','1','2','3','4',"[CLS]", "[SEP]"])
    return self.labels



  def _create_examples(self, df, set_type):
    """Creates examples for the training and dev sets."""

    examples = []

    for (i, line) in df.iterrows():
       
      # if i == 0:
      #   continue
      guid = "%s-%s" % (set_type, i)
      texts = tokenization.convert_to_unicode(line['texts'])
      label = tokenization.convert_to_unicode(line['labels'])
     
      examples.append(
          InputExample(guid=guid, text=texts, label=label))
    return examples


  def submit(self,texts,label_preds):
    df_test = self.test_df 
    print(df_test.shape)
    
    print(df_test['labels'].values[0])
    print(label_preds[0])
    res = decoder(label_preds,texts,FLAGS.label_type)


    logging.info("res[0]:%s"%(res[0]))
    logging.info('submit res len:%d'%len(res))
    df_test['entity_pred']=res

    #聚合结果
    df_test=postposs(df_test)

    df_test['entity_sub'] = df_test['entity_pred'].map(lambda x:post(x))

    logging.info('entity_sub:%d'%len(res))

    logging.info('predict res:')
    



    df_test['entity_sub'] = df_test['entity_sub'].map(lambda x:[i for i in x if i!=''])

    #print(df_test['entity_sub'].map(lambda x:len(x)))

    df_test['emotion_pred']=df_test.apply(lambda x:['POS']*len(x['entity_sub']),axis=1)
    

    df_test['entity_sub'] = df_test['entity_sub'].map(lambda x:','.join(x))
    df_test['emotion_pred'] = df_test['emotion_pred'].map(lambda x:','.join(x))


    if FLAGS.title_only:
        print(df_test[['entity','title','entity_sub']].head())

    else:
        print(df_test[['entity','entity_sub']].head())


    if FLAGS.testisdev or FLAGS.trainisdev:

      reals = df_test.entity.values
      preds = df_test.entity_sub.values
      p,r,f1=score(reals,preds)
      output_predict_file = os.path.join(FLAGS.output_dir, "bert_dev_ortrain.txt")
      
      df_test[['newsId','entity','entity_sub','entity_pred']].to_csv(output_predict_file, sep='\t',index=False,header=False)

    else:
      
      output_predict_file = os.path.join(FLAGS.output_dir, "bert_res.txt")
      
      path = FLAGS.root_path+'/data/coreEntityEmotion_sample_submission_v2.txt'
      df1 =pd.read_csv(path,names=['newsId','entity','emotion'],sep='\t')


        

      df_test = pd.merge(df1[['newsId']],df_test[['newsId','entity_sub','emotion_pred','entity_pred']],how='left')

      df_test[['newsId','entity_sub','entity_pred','emotion_pred']].to_csv(output_predict_file+'.all_pred.csv',sep='\t',index=False,header=False)

      df_test[['newsId','entity_sub','emotion_pred']].to_csv(output_predict_file, sep='\t',index=False,header=False)

      logging.info('test_predict done! and save in %s'%(output_predict_file))


# class NerProcessor(DataProcessor):
#     def __init__(self, output_dir):
#         self.labels = set()
#         self.output_dir = output_dir

#     def get_train_examples(self, data_dir):
#         return self._create_example(
#              get_train_data(), "train"
#         )

#     def get_dev_examples(self, data_dir):
#         return self._create_example(
#            get_dev_data(), "dev"
#         )

#     def get_test_examples(self, data_dir):
# #         return self._create_example(
# #             self._read_data(os.path.join(data_dir, "test.txt")), "test")
    
#         return self._create_example(
#             get_test_data(), "test")

#     def get_labels(self, labels=None):
    
#         #self.labels =# set(["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"])
#         self.labels = set(['0','1','2','3',"[CLS]", "[SEP]"])
#         return self.labels

#     def _create_example(self, lines, set_type):
#         examples = []
#         logging.error('lines[0]')

#         print(lines[0])
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             text = tokenization.convert_to_unicode(line[1])
#             label = tokenization.convert_to_unicode(line[0])
#             examples.append(InputExample(guid=guid, text=text, label=label))
#         return examples

#     def _read_data(self, input_file):
#         """Reads a BIO data.

#         lines 
#         [
#         ['O  O O O B-LOC B-LOC O B-LOC B- O O', '你 好 啊 啊 啊 啊 '],
#         ['O  O O O B-LOC B-LOC O B-LOC B- O O', '你 好 啊 啊 啊 啊 ']
#         ]
#         """
        
#         lines =[]
#         return lines


def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text.split(',')
    labellist = example.label.split(',')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("0")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'的时候才有效
    write_tokens(ntokens, output_dir, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params,):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels,FLAGS.crf_only, FLAGS.dropout_rate, FLAGS.lstm_size, FLAGS.cell, FLAGS.num_layers,)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                 modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 打印变量名
        # logger.info("**** Trainable Variables ****")
        #
        # # 打印加载模型的参数
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     logger.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            #train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                 total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=FLAGS.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids,):

               
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                 

                }

            eval_metrics = metric_fn(label_ids, pred_ids,)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


# def load_data():
#     processer = NerProcessor()
#     processer.get_labels()
#     example = processer.get_train_examples(FLAGS.data_dir)
#     print()

def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        tf.logging.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path: 
    :return: 
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_map)


    processors = {
       # "ner": NerProcessor,
        "sohuner":SOHUNERProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    #check output dir exists
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    processor = processors[FLAGS.ner]()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=session_config
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train and FLAGS.do_eval:
        # 加载训练数据
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) *1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        # 打印验证集数据信息
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

    label_list = processor.get_labels()
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    
        )

    params = {
        'batch_size': FLAGS.batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if FLAGS.do_train and FLAGS.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, FLAGS.output_dir)

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, FLAGS.output_dir)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # train and eval togither
        # early stop hook
        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=FLAGS.save_checkpoints_steps)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        
        
        if FLAGS.testisdev:
            predict_file = os.path.join(FLAGS.output_dir, "predict_dev.tf_record")
            filed_based_convert_examples_to_features(predict_examples, label_list,
                                                             FLAGS.max_seq_length, tokenizer,
                                                         predict_file, FLAGS.output_dir, mode="test")
        else:
            predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

            if not os.path.exists(predict_file) :
                filed_based_convert_examples_to_features(predict_examples, label_list,
                                                             FLAGS.max_seq_length, tokenizer,
                                                         predict_file, FLAGS.output_dir, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        
        texts ,label_preds = [],[]
        
        for predict_line, prediction in zip(predict_examples, result):
            idx = 0
            line = ''
            line_token = str(predict_line.text).split(' ')
            label_token = str(predict_line.label).split(' ')
            
            text ,label_pred = [],[]
            
            len_seq = len(label_token)
            if len(line_token) != len(label_token):
                tf.logging.info(predict_line.text)
                tf.logging.info(predict_line.label)
                break
            for id in prediction:
                if idx >= len_seq:
                    break
                if id == 0:
                    continue
                curr_labels = id2label[id]
                if curr_labels in ['[CLS]', '[SEP]']:
                    continue
                try:
                    line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    
                    text.append(line_token[idx])
                    label_pred.append(int(curr_labels))
                    
                except Exception as e:
                    tf.logging.info(e)
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                    line = ''
                    break
                idx += 1
            texts.append(text)
            label_preds.append(label_pred)
        processor.submit(texts,label_preds,)



        # with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
        #     result_to_pair(writer)
#         from bert_base.train import conlleval
#         eval_result = conlleval.return_report(output_predict_file)
#         print(''.join(eval_result))
#         # 写结果到文件中
#         with codecs.open(os.path.join(FLAGS.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
#             fd.write(''.join(eval_result))
    # filter model
    if FLAGS.filter_adam_var:
        adam_filter(FLAGS.output_dir)
if __name__ == "__main__":
  print(FLAGS.vocab_file)

  tf.app.run()

