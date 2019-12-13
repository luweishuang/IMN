import os, sys
import json
import random
from random import randint
import tensorflow as tf
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import data_helpers

repo_dir = os.path.dirname(cur_dir)
DATA_DIR = os.path.join(repo_dir, "data/Ecommerce_Corpus")

# Files
tf.flags.DEFINE_string("response_file", os.path.join(DATA_DIR, "candidate_responses_ecommerce.txt"), "path to response file")
tf.flags.DEFINE_string("vocab_file", os.path.join(DATA_DIR, "vocab.txt"), "vocabulary file")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_utter_len", 50, "max utterance length")
tf.flags.DEFINE_integer("max_utter_num", 10, "max utterance number")
tf.flags.DEFINE_integer("max_response_len", 50, "max response length")
tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoints", "checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

word2id = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(word2id)))

response_data = []
with open(FLAGS.response_file, 'rt') as f:
    for line in f:
        response_data.append(line.strip())
'''
user: 货要 真的
system:正品 有 保障 的 哦 亲亲 放心 呢
user:好 的
system:谢谢您 对 我 和 我们 店铺 的 信赖 我们 时刻 等待 着 您 的 再次 光临 哦 祝您 生活 愉快
'''
test_dialogue_data = json.load(open(os.path.join(DATA_DIR, "all_test_dialogue.json"), "r", encoding="utf-8"))
random.shuffle(test_dialogue_data)

target_loss_weight = [1.0, 1.0]

print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        utterances = graph.get_operation_by_name("utterances").outputs[0]
        response   = graph.get_operation_by_name("response").outputs[0]

        utterances_len = graph.get_operation_by_name("utterances_len").outputs[0]
        response_len = graph.get_operation_by_name("response_len").outputs[0]
        utterances_num = graph.get_operation_by_name("utterances_num").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        prob = graph.get_operation_by_name("prediction_layer/prob").outputs[0]

        query_sent = ""
        cur_index = randint(0, len(test_dialogue_data))
        demo_dialogue = test_dialogue_data[cur_index]
        print("--------------true dialogue begin------------------------")
        for idx, cur_line in demo_dialogue.items():
            if int(idx) % 2 == 0:
                print("user: " + cur_line)
            else:
                print("agent: " + cur_line)
        print("--------------true dialogue end------------------------")
        while True:
            user_input = input("user:")
            if user_input == "quit":
                break
            query_sent += user_input.strip()
            # print("query_sent: " + query_sent)
            test_dataset = data_helpers.load_dataset_infer(query_sent, word2id, FLAGS.max_utter_len, FLAGS.max_utter_num, response_data, FLAGS.max_response_len)
            # print('test_pairs: {}'.format(len(test_dataset)))

            rst_scores = np.zeros((len(test_dataset), ), dtype='float32')
            num_test = 0
            test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, FLAGS.max_response_len, shuffle=False)
            for test_batch in test_batches:
                x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs = test_batch
                feed_dict = {
                    utterances: x_utterances,
                    response: x_response,
                    utterances_len: x_utterances_len,
                    response_len: x_response_len,
                    utterances_num: x_utters_num,
                    dropout_keep_prob: 1.0
                }
                predicted_prob = sess.run(prob, feed_dict)

                rst_scores[num_test:num_test + len(predicted_prob)] = predicted_prob
                num_test += len(predicted_prob)
                # print('num_test_sample={}'.format(num_test))
            rst_scores_list = rst_scores.tolist()
            # max_score = max(rst_scores_list)  # 返回最大值
            # max_index = rst_scores_list.index(max_score)  # 返回最大值的索引
            # print("max_index=%d, max_score=%f " % (max_index, max_score))
            # print(response_data[max_index])
            indies = np.argsort(-rst_scores)
            rst_scores_list.sort(reverse=True)
            print("agent: " + response_data[indies[0]])
            for jj in range(5):
                print(str(rst_scores_list[jj]) + "==> " + response_data[indies[jj]])
            query_sent += " _EOS_" + response_data[indies[0]] + " _EOS_"
