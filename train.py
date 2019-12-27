import json
import os
import argparse
import tensorflow as tf
from transformer_model import TransformerModel
from data_loader import mk_lm_pny_vocab, mk_lm_han_vocab, process_file, read_file, next_batch

class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), args.config_path), "r") as fr:
            self.config = json.load(fr)
        print(self.config)
        self.load_data()
        self.word_vectors = None
        self.model = TransformerModel(config=self.config, vocab_size=self.vocab_size, label_size=self.label_size,
                                      word_vectors=self.word_vectors)

    def load_data(self):
        # 加载数据集
        train_path = 'data/train.tsv'
        dev_path = 'data/dev.tsv'
        test_path = 'data/test.tsv'
        pny_list, han_list = read_file(train_path)
        pny_dict_w2id, pny_dict_id2w = mk_lm_pny_vocab(pny_list)
        han_dict_w2id, han_dict_id2w = mk_lm_han_vocab(han_list)

        self.train_inputs, self.train_labels = process_file(train_path, pny_dict_w2id, han_dict_w2id,
                                                            self.config['sequence_length'])
        self.eval_inputs, self.eval_labels = process_file(dev_path, pny_dict_w2id, han_dict_w2id,
                                                          self.config['sequence_length'])

        self.vocab_size = len(pny_dict_w2id)
        self.label_size = len(han_dict_w2id)

    def train(self):

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                              self.config["output_path"] + "/summary/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                             self.config["output_path"] + "/summary/eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in next_batch(self.train_inputs, self.train_labels,
                                        self.config["batch_size"]):
                    summary, loss, acc = self.model.train(sess, batch, self.config["keep_prob"])
                    train_summary_writer.add_summary(summary)
                    print("train: step: {}, loss: {}, acc: {}".format(current_step, loss, acc))
                    current_step += 1

                    if current_step % self.config["checkpoint_every"] == 0:
                        eval_losses = []
                        for eval_batch in next_batch(self.eval_inputs, self.eval_labels,
                                                     self.config["batch_size"]):
                            eval_summary, eval_loss, acc = self.model.eval(sess, eval_batch)
                            eval_summary_writer.add_summary(eval_summary)


                            eval_losses.append(eval_loss)
                        print("\n")
                        print("eval:  loss: {} ,acc: {}".format(tf.reduce_mean(eval_losses), acc))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

            # inputs = {"inputs": tf.saved_model.utils.build_tensor_info(self.model.inputs),
            #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.model.keep_prob)}
            #
            # outputs = {"predictions": tf.saved_model.utils.build_tensor_info(self.model.predictions)}
            #
            # # method_name决定了之后的url应该是predict还是classifier或者regress
            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
            #                                                                               outputs=outputs,
            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            # self.builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
            #                                           signature_def_map={"classifier": prediction_signature},
            #                                           legacy_init_op=legacy_init_op)
            #
            # self.builder.save()


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    path = r'D:\anaconda\WORKSPACE\DeepSpeechRecognition-master\DeepSpeechRecognition-master\model_language\transformer_config.json'
    parser.add_argument("--config_path", help="config path of model", default=path)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
