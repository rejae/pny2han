import os
import tensorflow as tf
from data_loader import read_file, mk_lm_han_vocab, mk_lm_pny_vocab, next_batch, load_data
from utils import get_data, data_hparams

import warnings

warnings.filterwarnings('ignore')


# def load_data(train=True):
#     # 加载数据集
#     train_path = 'data/train.tsv'
#     dev_path = 'data/dev.tsv'
#     test_path = 'data/test.tsv'
#     train_pny_list, train_han_list = read_file(train_path)
#     pny_dict_w2id, pny_dict_id2w = mk_lm_pny_vocab(train_pny_list)
#     han_dict_w2id, han_dict_id2w = mk_lm_han_vocab(train_han_list)
#
#     dev_pny_list, dev_han_list = read_file(dev_path)
#
#     dev_pny_list, dev_han_list = read_file(test_path)
#     # train_inputs, train_labels = process_file(train_path, pny_dict_w2id, han_dict_w2id,
#     #                                           max_len)
#     # eval_inputs, eval_labels = process_file(dev_path, pny_dict_w2id, han_dict_w2id,
#     #                                         max_len)
#     vocab_size = len(pny_dict_w2id)
#     label_size = len(han_dict_w2id)
#     if train:
#         return train_pny_list, train_han_list, vocab_size, label_size, dev_pny_list, dev_han_list, pny_dict_w2id, han_dict_w2id
#     else:
#         return dev_pny_list, dev_han_list


# 0.准备训练所需数据----data_hparams 参数修改--------------------------
# data_args = data_hparams()
# data_args.data_type = 'train'
# train_data = get_data(data_args)
#
# # 0.准备验证所需数据  data_hparams   ------------------------------
# data_args = data_hparams()
# data_args.data_type = 'dev'
# dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------

# 2.语言模型训练-------------------------------------------
from transformer import Lm, lm_hparams

train_inputs, train_labels, input_vb_size, label_vb_size, eval_inputs, eval_labels, pny_dict_w2id, han_dict_w2id = load_data()

lm_args = lm_hparams(input_vb_size, label_vb_size)
lm = Lm(lm_args)

batch_num = len(train_inputs) // lm_args.batch_size
epochs = 1

with lm.graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0

    if os.path.exists('logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())

    print('enter epoch training')
    for epoch in range(epochs):

        total_loss = []
        total_acc = []
        i = 0
        for batch in next_batch(train_inputs, train_labels, lm_args.batch_size, pny_dict_w2id, han_dict_w2id):
            input_batch, label_batch = batch['x'], batch['y']
            try:
                feed = {lm.x: input_batch, lm.y: label_batch}
                acc, cost, _ = sess.run([lm.acc, lm.mean_loss, lm.train_op], feed_dict=feed)
                total_acc.append(acc)
                total_loss.append(cost)
                i = i + 1
            except Exception as e:
                print(e, 'batch_num:', i)
                print(input_batch, label_batch)

            if i % 300 == 0:
                print('acc=', sum(total_acc[-100:]) / 100, 'cost=', sum(total_loss[-100:]) / 100)
                ## evaluate
                eval_total_acc = []
                eval_total_loss = []
                j = 0
                for batch in next_batch(eval_inputs, eval_labels, lm_args.batch_size, pny_dict_w2id, han_dict_w2id):

                    input_batch, label_batch = batch['x'], batch['y']
                    feed = {lm.x: input_batch, lm.y: label_batch}
                    acc, cost, _ = sess.run([lm.acc, lm.mean_loss, lm.train_op], feed_dict=feed)
                    eval_total_acc.append(acc)
                    eval_total_loss.append(cost)
                    j = j + 1
                    if j == 100:
                        print('eval_total_acc:', sum(eval_total_acc) / (len(eval_total_acc)), 'eval_total_loss:',
                              sum(eval_total_loss) / (len(eval_total_loss)))
                        break

            if (epoch * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, epoch * batch_num + i)
        print('batch ', i + 1, ': average loss = ', sum(total_loss) / batch_num, 'average acc = ',
              sum(total_acc) / batch_num)
    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
    writer.close()
