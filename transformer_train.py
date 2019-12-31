import os
import tensorflow as tf
from data_loader import next_batch, load_data
import warnings
from transformer import Lm, lm_hparams
warnings.filterwarnings('ignore')

train_inputs, train_labels, pny_dict_w2id, han_dict_w2id = load_data()

input_vb_size = len(pny_dict_w2id)
label_vb_size = len(han_dict_w2id)

lm_args = lm_hparams(input_vb_size, label_vb_size)
lm = Lm(lm_args)

batch_num = len(train_inputs) // lm_args.batch_size
epochs = 5

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

            if (epoch * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, epoch * batch_num + i)
        print('batch ', i + 1, ': average loss = ', sum(total_loss) / batch_num, 'average acc = ',
              sum(total_acc) / batch_num)
    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
    writer.close()
