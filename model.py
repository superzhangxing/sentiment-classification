# -*- coding: utf-8 -*-

import tensorflow as tf
import dataset
from func import rnn_lstm,rnn_bi_lstm,embedding_dropout

# SENTENCE_LENGTH = 100
# BATCH_SIZE = 25
# HIDDEN = 100
# EMBEDDING_SIZE = 300

# class Model(object):
#     def __init__(self,batch,embedding_mat,trainable):
#         self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
#                                            initializer=tf.constant_initializer(0), trainable=False)
#         self.c,self.y = batch.get_next()
#         self.is_train = tf.get_variable(
#             "is_train", shape=[], dtype=tf.bool, trainable=False)
#         self.embedding_mat = tf.get_variable("embedding_mat", initializer=tf.constant(
#             embedding_mat, dtype=tf.float32), trainable=False)
#
#         self.c_mask = tf.cast(self.c, tf.bool)
#         self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
#
#         self.c_maxlen = SENTENCE_LENGTH
#
#         self.ready()
#
#         if trainable:
#             self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
#             self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
#
#
#
#     def ready(self):
#         N = BATCH_SIZE
#         CL = self.c_maxlen
#         H = HIDDEN
#
#         with tf.name_scope("word"):
#             c_emb = tf.nn.embedding_lookup(self.embedding_mat,self.c)
#
#         with tf.variable_scope("encoding"):
#             rnn = func.lstm(num_layers=3, num_units=H, batch_size=N, input_size=EMBEDDING_SIZE, keep_prob=1.0,
#                                 is_train=self.is_train)
#             output = rnn(inputs=c_emb, seq_len=self.c_len)
#
#         with tf.variable_scope("predict"):
#             W_fc = tf.get_variable("W_fc",[N,H], dtype=tf.float32, initializer=tf.random_uniform([N,H],maxval=1.0/H))
#             b_fc = tf.get_variable("b_fc",[N], dtype=tf.float32, initializer=tf.zeros([N]))
#             logits = tf.nn.bias_add(tf.matmul(W_fc,output), b_fc)
#             losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=tf.stop_gradient(logits))
#             self.loss = tf.reduce_mean(losses)


class Model(object):

    def __init__(self, config, is_training, c, y, keep_prob, embedding_mat):
        self.is_training = is_training
        self.c = c
        self.y = y
        self.keep_prob = keep_prob
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes

        self._calculate_sentence_length()

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding", dtype = tf.float32,
                                             initializer=tf.constant(embedding_mat,dtype=tf.float32),
                                             trainable=True)
            # self.embedding = tf.get_variable("embedding", dtype = tf.float32,
            #                                  initializer=tf.random_uniform([len(embedding_mat),len(embedding_mat[0])], dtype=tf.float32),
            #                                  trainable= True)
            inputs = tf.nn.embedding_lookup(self.embedding, self.c)

        inputs = embedding_dropout(inputs, self.keep_prob, is_train=self.is_training)
        # outputs1,state = self._build_rnn_graph_lstm(inputs=inputs, config=config, is_training=True)
        # rnn = rnn_lstm(num_layers=config.num_layers, hidden_size=config.hidden_size, batch_size=config.batch_size)
        # outputs1, state = rnn(inputs=inputs, sequence_length=self.c_len)
        rnn = rnn_bi_lstm(num_layers=config.num_layers, hidden_size=config.hidden_size, batch_size=config.batch_size)
        outputs1, state = rnn(inputs=inputs, sequence_length=self.c_len)

        # outputs_shape = tf.shape(outputs1)
        # outputs = tf.slice(outputs1,[0,outputs_shape[1]-1,0],[outputs_shape[0],1,outputs_shape[2]])
        # outputs = tf.reshape(outputs, [self.batch_size,-1])
        # outputs = state[-1][-1]

        outputs = tf.concat([state[0][-1][-1],state[-1][-1][-1]],axis=1)

        w = tf.get_variable("w", dtype=tf.float32,
                            initializer=tf.random_uniform([self.hidden_size*2,self.num_classes]))
        b = tf.get_variable("b", dtype=tf.float32, initializer=tf.zeros([self.batch_size, self.num_classes]))
        logits = tf.matmul(outputs,w)+b

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self.y), logits=logits)
        self.loss = tf.reduce_mean(losses)
        self.w = w
        self.b = b
        self.logits = logits
        self.losses = losses
        self.outputs = outputs1
        self.state = state

        # accuracy
        self.predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(self.y, axis=1)
        equals = tf.cast(tf.equal(self.predictions,labels),tf.int32)
        self.accuracy = tf.divide(tf.reduce_sum(equals),self.batch_size)

        if is_training:
            self.lr = tf.get_variable("lr", dtype=tf.float32,
                                      initializer=tf.constant(config.learning_rate,dtype=tf.float32), trainable=False)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        # cell
        cell = [tf.nn.rnn_cell.LSTMCell(config.hidden_size, state_is_tuple = True) for _ in range(config.num_layers)]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)

        initial_state = multi_rnn_cell.zero_state(self.batch_size,dtype=tf.float32)

        outputs,state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                          inputs=inputs,
                                          sequence_length= self.c_len,
                                          initial_state= initial_state,
                                          dtype=tf.float32)

        return outputs,state

    def _calculate_sentence_length(self):
        self.c_mask = tf.cast(self.c, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.c_max_len = tf.reduce_max(self.c_len)




