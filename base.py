import tensorflow as tf


class BaseModel(object):
    def __init__(self, config, vocab_size=None, label_size=None, word_vectors=None):

        self.config = config
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.word_vectors = word_vectors

        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")  # 数据输入
        self.labels = tf.placeholder(tf.int32, [None, None], name="labels")  # 标签
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.l2_loss = tf.constant(0.0)  # 定义l2损失
        self.loss = 0.0  # 损失
        self.train_op = None  # 训练入口
        self.summary_op = None
        self.logits = None  # 模型最后一层的输出
        self.acc = 0.0  # 预测结果
        self.saver = None  # 保存为ckpt模型的对象



    def cal_acc(self, preds, istarget):
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(preds, self.labels)) * istarget) / (tf.reduce_sum(istarget))
        tf.summary.scalar('acc', self.acc)
        return self.acc

    def cal_loss(self, y_smoothed, istarget):

        with tf.name_scope("loss"):

            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_smoothed)
            self.loss = tf.reduce_sum(self.loss * istarget) / (tf.reduce_sum(istarget))

            return self.loss

    def get_optimizer(self):
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.config["learning_rate"])
        return optimizer

    def get_train_op(self):
        # 定义优化器
        optimizer = self.get_optimizer()
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        tf.summary.scalar("loss", self.loss)

        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def get_predictions(self):
        logits_list = self.logits
        predictions = [tf.argmax(logit, axis=-1) for logit in logits_list]
        return predictions

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):

        feed_dict = {self.inputs: batch["x"],
                     self.labels: batch["y"],
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, summary, loss, acc = sess.run([self.train_op, self.summary_op, self.loss, self.acc],
                                                 feed_dict=feed_dict)
        return summary, loss, acc

    def eval(self, sess, batch):
        feed_dict = {self.inputs: batch["x"],
                     self.labels: batch["y"],
                     self.keep_prob: 1.0}

        summary, loss, acc = sess.run([self.summary_op, self.loss, self.acc], feed_dict=feed_dict)
        return summary, loss, acc

    def infer(self, sess, inputs):

        feed_dict = {self.inputs: inputs,
                     self.keep_prob: 1.0}

        predict = sess.run(self.acc, feed_dict=feed_dict)

        return predict
