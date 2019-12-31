# coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
import sys
import json
from data_loader import load_test_data
from collections import defaultdict
import warnings
from transformer import Lm, lm_hparams

warnings.filterwarnings('ignore')


# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取

# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


def defaultdict_from_dict(dic):
    dd = defaultdict(int)
    dd.update(dic)
    return dd


#  加载数据
test_pny_list, test_han_list = load_test_data()

# 1.声学模型-----------------------------------


# 2.语言模型-------------------------------------------拿到两个vocab的大小，来恢复模型，但是又重建了vocab，很浪费资源，所以需要保存下来

with open('vocab/pny_vocab.json', "r", encoding='utf-8') as f:
    pny_dict_w2id = json.load(f)
    pny_dict_w2id = defaultdict_from_dict(pny_dict_w2id)
pny_dict_id2w = {v: k for k, v in pny_dict_w2id.items()}

with open('vocab/han_vocab.json', "r", encoding='utf-8') as f:
    han_dict_w2id = json.load(f)
    han_dict_w2id = defaultdict_from_dict(han_dict_w2id)
han_dict_id2w = {v: k for k, v in han_dict_w2id.items()}

input_vb_size = len(pny_dict_w2id)
label_vb_size = len(han_dict_w2id)
lm_args = lm_hparams(input_vb_size, label_vb_size)
lm_args.dropout_rate = 0.
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver = tf.train.Saver()
with sess.as_default():
    latest = tf.train.latest_checkpoint('./logs_lm')
    saver.restore(sess, latest)

# 4. 进行测试-------------------------------------------
word_num = 0
word_error_num = 0

for i in range(len(test_pny_list)):
    print('\n the ', i, 'th example.')

    pny_id = [pny_dict_w2id[word] for word in test_pny_list[i]]
    han_id = [han_dict_w2id[word] for word in test_han_list[i]]

    with sess.as_default():

        pny_id = np.array(pny_id).reshape(1, -1)
        preds = sess.run(lm.preds, {lm.x: pny_id})
        # preds_reshape = np.reshape(preds, [1, -1])
        preds_list = preds.tolist()[0]

        result = ''.join([han_dict_id2w[idx] for idx in preds_list])
        label = test_han_list[i]
        print('原文汉字id:', ', '.join([str(han_dict_w2id[w]) for w in test_han_list[i]]))
        print('原文汉字：', test_han_list[i])

        print("识别结果id:", preds_list)
        print('识别结果汉字：：', result)
        distance = GetEditDistance(label, result)
        word_error_num += distance
        word_num += len(label)


print('词错误率：', word_error_num / word_num)
sess.close()
