# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
#from tqdm import tqdm

import dataset
from model import Model
from config import Config

MR_TRAIN_RECORD = 'dataset/MR/train_record'
EMBEDDING_MAT = 'dataset/MR/embedding_mat'
SUMMARY = 'summary'

MR_TRAIN_FILE = 'dataset/MR/train'
MR_DEV_FILE = 'dataset/MR/dev'

def parse(example):
    """ parse example which saved in records"""
    features = tf.parse_single_example(example,
                                        features={
                                            "feature": tf.FixedLenFeature([], tf.string),
                                            "label": tf.FixedLenFeature([], tf.int64)
                                        })
    feature = tf.decode_raw(features["feature"], tf.int64)
    label = features["label"]

    return feature, label

def train():
    ######################################################################
    # 加载词典
    ######################################################################
    with open(EMBEDDING_MAT,'rb') as fd:
        embedding_mat = pickle.load(fd)

    with open(MR_TRAIN_FILE,'rb') as fd:
        train_dataset = pickle.load(fd)

    with open(MR_DEV_FILE,'rb') as fd:
        dev_dataset = pickle.load(fd)


    config = Config()
    c = tf.placeholder(tf.int32, [config.batch_size, config.max_sentence_len])
    y = tf.placeholder(tf.float32, [config.batch_size, config.num_classes])
    model = Model(config = config, is_training=True, c=c, y=y , embedding_mat=embedding_mat)

    lr = 0.01

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SUMMARY)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(config.epochs):
            # train
            batchs = len(train_dataset) // config.batch_size
            for step in range(batchs):
                feed_c = train_dataset[step*config.batch_size:(step+1)*config.batch_size]
                feed_c = [c[:-1] for c in feed_c]
                feed_y_idx = train_dataset[step*config.batch_size:(step+1)*config.batch_size]
                feed_y = []
                label = []
                for i in range(len(feed_y_idx)):
                    feed_y.append([])
                    for j in range(config.num_classes):
                        feed_y[i].append(0)
                    feed_y[i][feed_y_idx[i][-1]] = 1
                    label.append(feed_y_idx[i][-1])

                loss,train_op,predictions,w,b,emb,outputs,state = sess.run([model.loss, model.train_op, model.predictions, model.w, model.b,model.embedding,model.outputs,model.state],
                                                     feed_dict={c:feed_c,y:feed_y})
                accuracy = 0.
                for i in range(config.batch_size):
                    if predictions[i] == label[i]:
                        accuracy += 1.
                accuracy = accuracy / config.batch_size

                if step % 10 == 0:
                    print('epoch: {},step: {}, loss: {:.2f}, accuracy: {:.2f}'.format(epoch,step,loss,accuracy))

            # dev
            batchs = len(dev_dataset) // config.batch_size
            total_accuracy = 0.
            for step in range(batchs):
                feed_c = dev_dataset[step*config.batch_size:(step+1)*config.batch_size]
                feed_c = [c[:-1] for c in feed_c]
                feed_y_idx = dev_dataset[step*config.batch_size:(step+1)*config.batch_size]
                feed_y = []
                label = []
                for i in range(len(feed_y_idx)):
                    feed_y.append([])
                    for j in range(config.num_classes):
                        feed_y[i].append(0)
                    feed_y[i][feed_y_idx[i][-1]] = 1
                    label.append(feed_y_idx[i][-1])

                loss,predictions= sess.run([model.loss,model.predictions],feed_dict={c:feed_c,y:feed_y})

                accuracy = 0.
                for i in range(config.batch_size):
                    if predictions[i] == label[i]:
                        accuracy += 1.
                accuracy = accuracy / config.batch_size
                total_accuracy += accuracy
            total_accuracy = total_accuracy / batchs
            print('epoch: {}, dev accuracy: {:.2f}'.format(epoch,total_accuracy))




train()