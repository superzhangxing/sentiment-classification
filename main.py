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

MR_LEXICON = 'dataset/MR/lexicon'

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

    with open(MR_LEXICON,'rb') as fd:
        lexicon = pickle.load(fd)

    with open(MR_TRAIN_FILE,'rb') as fd:
        train_dataset = pickle.load(fd)

    with open(MR_DEV_FILE,'rb') as fd:
        dev_dataset = pickle.load(fd)


    config = Config()
    c = tf.placeholder(tf.int32, [config.batch_size, config.max_sentence_len])
    y = tf.placeholder(tf.float32, [config.batch_size, config.num_classes])
    keep_prob = tf.placeholder(tf.float32, [config.batch_size, config.max_sentence_len])
    model = Model(config = config, is_training=True, c=c, y=y , keep_prob=keep_prob, embedding_mat=embedding_mat)

    lr = 0.01

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SUMMARY)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(config.epochs):
            # train
            model.is_training = True
            batchs = len(train_dataset) // config.batch_size
            total_loss = 0.
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
                feed_keep_prob  = []
                for i in range(len(feed_c)):
                    feed_keep_prob.append([])
                    for j in range(len(feed_c[0])):
                        id = feed_c[i][j]
                        if id in lexicon:
                            feed_keep_prob[i].append(config.sentiment_keep)
                        else:
                            feed_keep_prob[i].append(config.neural_keep)

                loss,train_op,predictions,w,b,emb,outputs,state,accuracy = sess.run([model.loss, model.train_op, model.predictions, model.w, model.b,model.embedding,model.outputs,model.state,model.accuracy],
                                                     feed_dict={c:feed_c,y:feed_y,keep_prob:feed_keep_prob})
                total_loss += loss*config.batch_size
                if step % 10 == 0:
                    print('epoch: {},step: {}, loss: {:.2f}, accuracy: {:.2f}'.format(epoch,step,loss,accuracy))
            print('epoch: {}, train loss: {:.2f}'.format(epoch, total_loss))

            # dev
            model.is_training = False
            batchs = len(dev_dataset) // config.batch_size
            total_accuracy = 0.
            total_loss = 0.
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

                feed_keep_prob  = []
                for i in range(len(feed_c)):
                    feed_keep_prob.append([])
                    for j in range(len(feed_c[0])):
                        id = feed_c[i][j]
                        if id in lexicon:
                            feed_keep_prob[i].append(config.sentiment_keep)
                        else:
                            feed_keep_prob[i].append(config.neural_keep)

                loss,predictions,accuracy= sess.run([model.loss,model.predictions,model.accuracy],
                                                    feed_dict={c:feed_c,y:feed_y,keep_prob:feed_keep_prob})
                total_loss += loss * config.batch_size

                total_accuracy += accuracy
            total_accuracy = total_accuracy / batchs
            print('epoch: {}, dev loss: {:.2f}'.format(epoch, total_loss))
            print('epoch: {}, dev accuracy: {:.3f}'.format(epoch,total_accuracy))

            save_path = saver.save(sess,"saver/model.ckpt")
            print("model saved in path : {}".format(save_path))




train()