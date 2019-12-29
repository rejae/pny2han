import os
import tensorflow as tf
from data_loader import read_file, mk_lm_han_vocab, mk_lm_pny_vocab, process_file, next_batch
from utils import get_data, data_hparams

import warnings

warnings.filterwarnings('ignore')

# def load_data(self):
#     # 加载数据集
#     train_path = 'data/train.tsv'
#     dev_path = 'data/dev.tsv'
#     test_path = 'data/test.tsv'
#     pny_list, han_list = read_file(train_path)
#     pny_dict_w2id, pny_dict_id2w = mk_lm_pny_vocab(pny_list)
#     han_dict_w2id, han_dict_id2w = mk_lm_han_vocab(han_list)
#
#     self.train_inputs, self.train_labels = process_file(train_path, pny_dict_w2id, han_dict_w2id)
#     self.vocab_size = len(pny_dict_w2id)

# 0.准备训练所需数据----data_hparams 参数修改--------------------------
data_args = data_hparams()
data_args.data_type = 'train'
train_data = get_data(data_args)

# 0.准备验证所需数据  data_hparams   ------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'

dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------

# 2.语言模型训练-------------------------------------------
from transformer import Lm, lm_hparams

input_vb_size = len(train_data.pny_vocab)
label_vb_size = len(train_data.han_vocab)
lm_args = lm_hparams(input_vb_size, label_vb_size)
lm = Lm(lm_args)

batch_num = len(train_data.pny_lst) // train_data.batch_size
epochs = 1

with lm.graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0

    if os.path.exists('logs_lm_improve/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm_improve')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('logs_lm_improve/tensorboard', tf.get_default_graph())

    print('enter epoch training')
    for epoch in range(epochs):

        total_loss = []
        total_acc = []
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)

            feed = {lm.x: input_batch, lm.y: label_batch}
            acc, cost, _ = sess.run([lm.acc, lm.mean_loss, lm.train_op], feed_dict=feed)
            total_acc.append(acc)
            total_loss.append(float(cost))
            i = i + 1
            if i % 300 == 0:

                print('train', 'average acc=', sum(total_acc[-100:]) / 100, 'cost=', sum(total_loss[-100:]) / 100)
                # dev_total_loss = []
                # dev_total_acc = []
                # batch = dev_data.get_lm_batch()
                # dev_batch_num = len(dev_data.pny_lst) // dev_data.batch_size
                # for j in range(dev_batch_num):
                #     input_batch, label_batch = next(batch)
                #
                #     feed = {lm.x: input_batch, lm.y: label_batch}
                #     acc, cost, _ = sess.run([lm.acc, lm.mean_loss, lm.train_op], feed_dict=feed)
                #     dev_total_acc.append(acc)
                #     dev_total_loss.append(float(cost))
                #     j = j + 1
                #     if j == 100:
                #         break
                # print('evaluate', 'average acc=', sum(dev_total_acc[-10:]) / 10, 'cost=',
                #       sum(dev_total_loss[-10:]) / 10)
            if (epoch * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, epoch * batch_num + i)
        print('batch ', i + 1, ': average loss = ', sum(total_loss) / batch_num, 'average acc = ',
              sum(total_acc) / batch_num)
