# -*- coding: utf-8 -*-

import tensorflow as tf

import pickle
import numpy as np
import random


MR_NEG_FILE = 'dataset/MR/rt-polarity.neg1'
MR_POS_FILE = 'dataset/MR/rt-polarity.pos1'

MR_TRAIN_RECORD = 'dataset/MR/train_record'
MR_DEV_RECORD = 'dataset/MR/dev_record'

MR_TRAIN_FILE = 'dataset/MR/train'
MR_DEV_FILE = 'dataset/MR/dev'

LEXICON = 'dataset/lexicon_negpos'
EMBEDDING_LENGTH = 300

GLOVE = 'dataset/glove.840B.300d.txt'
EMBEDDING_MAT = 'dataset/MR/embedding_mat'

DATASETS = ['MR','sst']

SENTENCE_LENGTH = 100

SPACE = '<spa>'
UNKNOWN = '<unk>'

class Dataset_MR(object):
    def __init__(self):
        self.vocab_word2id,self.vocab_id2word = self.generate_vocab()
        self.vocab_id2label = self.generate_lexicon()
        self.vocab_id2embedding = self.generate_embedding()
        self.generate_record()
        self.generate_train_dev_files()

    def generate_vocab(self):
        files = [MR_NEG_FILE,MR_POS_FILE]
        vocab_count = {}
        vocab_word2id = {}
        vocab_id2word = {}
        for file in files:
            with open(file,'r',encoding='utf-8') as fd:
                for line_id,line in enumerate(fd):
                    lst_line = line.split()
                    for word in lst_line:
                        vocab_count[word] = vocab_count.get(word,0) + 1

        sorted_vocab = sorted(vocab_count.items(), key=lambda item: item[1], reverse=True)

        id = 2
        vocab_id2word[0] = SPACE
        vocab_id2word[1] = UNKNOWN
        vocab_word2id[SPACE] = 0
        vocab_word2id[UNKNOWN] = 1
        for word_pair in sorted_vocab:
            vocab_id2word[id] = word_pair[0]
            vocab_word2id[word_pair[0]] = id
            id += 1

        print('vocab length: {}'.format(len(vocab_count)))
        return vocab_word2id,vocab_id2word

    def generate_lexicon(self):
        vocab_id2label = {}
        with open(LEXICON,'rb') as fd:
            lexicon = pickle.load(fd)
        for id,word in self.vocab_id2word.items():
            if(word in lexicon):
                vocab_id2label[id] = lexicon[word]

        print('lexicon length: {}'.format(len(vocab_id2label)))
        return vocab_id2label


    def generate_embedding(self):
        vocab_id2embedding = {}
        with open(GLOVE,'r',encoding='utf-8') as fd:
            for line_id, line in enumerate(fd):
                lst_line = line.split()
                word = "".join(lst_line[0:-EMBEDDING_LENGTH])
                vector = list(map(float, lst_line[-EMBEDDING_LENGTH:]))
                if(word in self.vocab_word2id):
                    vocab_id2embedding[self.vocab_word2id[word]] = vector

        for id in self.vocab_id2word:
            if(id not in vocab_id2embedding):
                vocab_id2embedding[id] = [np.random.normal(scale=0.01) for _ in range(EMBEDDING_LENGTH)]

        embedding_mat = [vocab_id2embedding[id] for id in range(len(vocab_id2embedding))]
        print('embedding dict length: {}'.format(len(vocab_id2embedding)))

        with open(EMBEDDING_MAT,'wb') as fd:
            pickle.dump(embedding_mat,fd)

        return vocab_id2embedding


    def generate_record(self):
        train_writer = tf.python_io.TFRecordWriter(MR_TRAIN_RECORD)

        with open(MR_NEG_FILE,'r',encoding='utf-8') as fd:
            label = 0
            for line_id, line in enumerate(fd):
                context_idxs = np.zeros([SENTENCE_LENGTH], dtype=np.int32)
                lst_line = line.split()
                for i,word in enumerate(lst_line):
                    context_idxs[i] = self.vocab_word2id[word]

                    record = tf.train.Example(features = tf.train.Features(
                        feature = {
                            "context_idxs":tf.train.Feature(bytes_list=tf.train.BytesList(value = [context_idxs.tostring()])),
                            "label":tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
                        }
                    ))

                    train_writer.write(record.SerializeToString())

        with open(MR_POS_FILE, 'r', encoding='utf-8') as fd:
            label = 1
            for line_id, line in enumerate(fd):
                context_idxs = np.zeros([SENTENCE_LENGTH], dtype=np.int32)
                lst_line = line.split()
                for i, word in enumerate(lst_line):
                    context_idxs[i] = self.vocab_word2id[word]

                    record = tf.train.Example(features=tf.train.Features(
                        feature={
                            "context_idxs": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                            "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
                        }
                    ))

                    train_writer.write(record.SerializeToString())

    def generate_train_dev_files(self):
        context_idx = []
        with open(MR_NEG_FILE, 'r', encoding='utf-8') as fd:
            for line_id, line in enumerate(fd):
                line_idx = np.zeros([SENTENCE_LENGTH+1], dtype=np.int32)
                lst_line = line.split()
                for i, word in enumerate(lst_line):
                    line_idx[i] = self.vocab_word2id[word]
                context_idx.append(line_idx)
        with open(MR_POS_FILE, 'r', encoding='utf-8') as fd:
            for line_id, line in enumerate(fd):
                line_idx = np.zeros([SENTENCE_LENGTH + 1], dtype=np.int32)
                lst_line = line.split()
                for i, word in enumerate(lst_line):
                    line_idx[i] = self.vocab_word2id[word]
                line_idx[-1] = 1
                context_idx.append(line_idx)

        random.shuffle(context_idx)
        dev = context_idx[:1000]
        train = context_idx[1000:]

        with open(MR_TRAIN_FILE,'wb') as fd:
            pickle.dump(train,fd)
        with open(MR_DEV_FILE, 'wb') as fd:
            pickle.dump(dev, fd)


# dataset_mr = Dataset_MR()
# with open(MR_DEV_FILE,'rb') as fd:
#     dev = pickle.load(fd)
#     print(dev)



