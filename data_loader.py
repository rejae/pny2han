import numpy as np
from collections import defaultdict
import random
import json


def read_file(filename):
    """读取文件数据"""

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        pny_list = []
        han_list = []
        for line in lines:
            pny_temp = []

            wav_file, pny, han = line.split('<SEP>')
            han = han.strip()
            for pny_item in pny.split(' '):
                pny_temp.append(pny_item)

            pny_list.append(pny_temp)
            han_list.append([item for item in han])

        han_list = [han for han in han_list]
    return pny_list, han_list


def mk_lm_pny_vocab(data):
    vocab = ['<PAD>']
    shuffle_vocab = []
    for line in data:
        for pny in line:
            if pny not in shuffle_vocab:
                shuffle_vocab.append(pny)
    random.shuffle(shuffle_vocab)
    vocab.extend(shuffle_vocab)
    pny_dict_w2id = defaultdict(int)
    for index, item in enumerate(vocab):
        pny_dict_w2id[item] = index

    with open('vocab/pny_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(pny_dict_w2id, f, ensure_ascii=False)
        f.write('\n')
    pny_dict_id2w = {v: k for k, v in pny_dict_w2id.items()}

    return pny_dict_w2id, pny_dict_id2w


def mk_lm_han_vocab(data):
    vocab = ['<PAD>']
    shuffle_vocab = []
    for line in data:
        # line = ''.join(line.split(' '))
        for han in line:
            if han not in shuffle_vocab:
                shuffle_vocab.append(han)

    random.shuffle(shuffle_vocab)
    vocab.extend(shuffle_vocab)

    han_dict_w2id = defaultdict(int)
    for index, item in enumerate(vocab):
        han_dict_w2id[item] = index

    han_dict_id2w = {v: k for k, v in han_dict_w2id.items()}

    with open('vocab/han_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(han_dict_w2id, f, ensure_ascii=False)
        f.write('\n')
    return han_dict_w2id, han_dict_id2w


def next_batch(x, y, batch_size, pnyw2id, han_w2id):
    num_batches = len(x) // batch_size

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = np.array(x)[perm]
    y = np.array(y)[perm]

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x = x.tolist()[start: end]
        batch_y = y.tolist()[start: end]

        max_len = max([len(line) for line in batch_x])
        input_batch = np.array(
            [[pnyw2id[item] for item in line] + [0] * (max_len - len(line)) for line in batch_x])
        label_batch = np.array(
            [[han_w2id[item] for item in line] + [0] * (max_len - len(line)) for line in batch_y])

        yield dict(x=input_batch, y=label_batch)


def load_data():
    # 加载数据集
    train_path = 'data/train.tsv'
    dev_path = 'data/dev.tsv'
    train_pny_list, train_han_list = read_file(train_path)
    dev_pny_list, dev_han_list = read_file(dev_path)

    pny_dict_w2id, pny_dict_id2w = mk_lm_pny_vocab(train_pny_list)
    han_dict_w2id, han_dict_id2w = mk_lm_han_vocab(train_han_list)

    return train_pny_list, train_han_list, dev_pny_list, dev_han_list, pny_dict_w2id, han_dict_w2id


def load_test_data():
    test_path = 'data/test.tsv'
    test_pny_list, test_han_list = read_file(test_path)

    return test_pny_list, test_han_list
