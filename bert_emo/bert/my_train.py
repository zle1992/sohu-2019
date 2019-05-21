import logging  
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # 
# Set the output directory for saving model file


from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


# Optionally, set a GCP bucket location

OUTPUT_DIR = 'OUTPUT_DIR_NAME'#@param {type:"string"}
#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False #@param {type:"boolean"}
#@markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.
USE_BUCKET = False #@param {type:"boolean"}
BUCKET = 'BUCKET_NAME' #@param {type:"string"}

if USE_BUCKET:
  OUTPUT_DIR = 'gs://{}/{}'.format(BUCKET, OUTPUT_DIR)
  from google.colab import auth
  auth.authenticate_user()

if DO_DELETE:
  try:
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  except:
    # Doesn't matter if the directory didn't exist
    pass
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))





from tensorflow import keras
import os
import re

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
def train(train,test,label_list):
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                       text_a = x['texta'], 
                                                                       text_b = x['textb'], 
                                                                       label = x['label']), axis = 1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                       text_a = x['texta'], 
                                                                       text_b = x['textb'],
                                                                       label = x['label']), axis = 1)


    parm={
    'vocab_file':"/sdb1/zhangle/2019/zhangle11/bert/chinese_L-12_H-768_A-12/vocab.txt",
    'bert_config_file':"/sdb1/zhangle/2019/zhangle11/bert/chinese_L-12_H-768_A-12/bert_config.json",
    'BATCH_SIZE' :32,
    'NUM_TRAIN_EPOCHS' :10.0,
     'MAX_SEQ_LENGTH':64,
    }

    tokenizer = tokenization.FullTokenizer(vocab_file=parm['vocab_file'])


    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = parm['MAX_SEQ_LENGTH']
    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


    import modeling
    def create_model(is_predicting, input_ids, input_mask, segment_ids,
                     labels, num_labels):
      """Creates a classification model."""
      model = modeling.BertModel(
          config = modeling.BertConfig.from_json_file(parm['bert_config_file']),
          is_training= not is_predicting,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids28 minutes ago
,
          use_one_hot_embeddings=True)

      # In the demo, we are doing a simple classification task on the entire
      # segment.
      #
      # If you want to use the token-level output, use model.get_sequence_output()
      # instead.
      output_layer = model.get_pooled_output()

      hidden_size = output_layer.shape[-1].value

      output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))

      output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

      with tf.variable_scope("loss"):
        if not is_predicting:
          # I.e., 0.1 dropout
          output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

        if is_predicting:
            return (predicted_labels, log_probs)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, predicted_labels,log_probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

          (loss, predicted_labels, log_probs) = create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

          train_op = bert.optimization.create_optimizer(
              loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

          # Calculate evaluation metrics. 
          def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            f1_score = tf.contrib.metrics.f1_score(
                label_ids,
                predicted_labels)
            auc = tf.metrics.auc(
                label_ids,
                predicted_labels)
            recall = tf.metrics.recall(
                label_ids,
                predicted_labels)
            precision = tf.metrics.precision(
                label_ids,
                predicted_labels) 
            true_pos = tf.metrics.true_positives(
                label_ids,
                predicted_labels)
            true_neg = tf.metrics.true_negatives(
                label_ids,
                predicted_labels)   
            false_pos = tf.metrics.false_positives(
                label_ids,
                predicted_labels)  
            false_neg = tf.metrics.false_negatives(
                label_ids,
                predicted_labels)
            return {
                "eval_accuracy": accuracy,
                "f1_score": f1_score,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "true_negatives": true_neg,
                "false_positives": false_pos,
                "false_negatives": false_neg
            }

          eval_metrics = metric_fn(label_ids, predicted_labels)

          if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=loss,
              train_op=train_op)
          else:
              return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
          (predicted_labels, log_probs) = create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

          predictions = {
              'probabilities': log_probs,
              'labels': predicted_labels
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # Return the actual model function in the closure
      return model_fn



    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = parm['BATCH_SIZE']
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = parm['NUM_TRAIN_EPOCHS']
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100


    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)



    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)


    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)


    estimator.evaluate(input_fn=test_input_fn, steps=None)

def submit():
    def getPrediction(in_sentences):
        labels = ["Negative", "Positive"]
        input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
        input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
        predictions = estimator.predict(predict_input_fn)
        return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]


    pred_sentences = [
      "That movie was absolutely awful",
      "The acting was a bit lacking",
      "The film was creative and surprising",
      "Absolutely fantastic!"
    ]

    predictions = getPrediction(pred_sentences)


def read_train():
    import pandas as pd
    df = pd.read_csv('/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_train.txt.pick.emotion_1_pair.csv',nrows=1000)
    kk = int(len(df)*0.001)
    print('read____________done!')
    return df[:kk],df[kk:]
def run_train():
    train_df, dev_df = read_train()
    
    label_list = [0, 1]
    train(train_df,dev_df,label_list)
def main():

    run_train()
if __name__ == '__main__':
    main()