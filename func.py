# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.ops.math_ops import floor
# import tensorflow.python.ops.nn_ops as nn_ops
# import tensorflow.python.ops.random_ops as random_ops
# import tensorflow.python.ops.math_ops as math_ops


# class lstm(object):
#     def __init__(self,num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="lstm"):
#         self.num_layers = num_layers
#         self.units = []
#         self.inits = []
#         self.dropout_mask = []
#         self.scope = scope
#
#         for layer in range(num_layers):
#             input_size_ = input_size if layer == 0 else num_units
#             unit = tf.contrib.rnn.LSTMCell(num_units)
#             init = tf.tile(tf.Variable(tf.zeros([1, num_units])), [batch_size, 1])
#             mask = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32), keep_prob=keep_prob,
#                            is_train=is_train, mode='recurrent')
#
#             self.inits.append(init)
#             self.units.append(unit)
#             self.dropout_mask.append(mask)
#
#     def __call__(self,inputs,seq_len, keep_prob=1.0, is_train=None,cancat_layers = True):
#         outputs = [inputs]
#         with tf.variable_scope(self.scope):
#             for layer in range(self.num_layers):
#                 unit = self.units[layer]
#                 init = self.inits[layer]
#                 mask = self.dropout_mask[layer]
#
#                 out,final_state = tf.nn.dynamic_rnn(unit,outputs[-1]*mask, seq_len, initial_state=init, dtype=tf.float32)
#
#                 outputs.append(out, axis = 2)
#
#         return outputs[-1]
#
#
# def dropout(args,keep_prob,is_train,mode='recurrent'):
#     if keep_prob < 1.0:
#         noise_shape = None
#         scale = 1.0
#         shape = tf.shape(args)
#
#         if mode == 'embedding':
#             noise_shape = [shape[0], 1]
#             scale = keep_prob
#         if mode == 'recurrent' and len(args.get_shape().as_list()) == 3:
#             noise_shape = args.get_shape()
#         args = tf.cond(is_train, lambda:tf.nn.dropout(
#             args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
#     return args

class rnn_lstm(object):
    def __init__(self,num_layers,hidden_size,batch_size):
        self.num_layers = num_layers
        self.hidded_size = hidden_size
        self.batch_size = batch_size

        cell = [tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True) for _ in range(num_layers)]
        self.multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
        self.initial_state = self.multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)

    def __call__(self, inputs,sequence_length):
        outputs, state = tf.nn.dynamic_rnn(cell=self.multi_rnn_cell,
                                           inputs=inputs,
                                           sequence_length=sequence_length,
                                           initial_state=self.initial_state,
                                           dtype=tf.float32)
        return outputs,state

class rnn_bi_lstm(object):
    def __init__(self,num_layers,hidden_size,batch_size):
        self.num_layers = num_layers
        self.hidded_size = hidden_size
        self.batch_size = batch_size

        fw_cell = [tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True) for _ in range(num_layers)]
        bw_cell = [tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True) for _ in range(num_layers)]
        self.fw_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell)
        self.fw_initial_state = self.fw_multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)
        self.bw_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell)
        self.bw_initial_state = self.bw_multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)

    def __call__(self, inputs,sequence_length):
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_multi_rnn_cell,
                                                         cell_bw=self.bw_multi_rnn_cell,
                                                         inputs=inputs,
                                                         sequence_length=sequence_length,
                                                         initial_state_fw=self.fw_initial_state,
                                                         initial_state_bw=self.bw_initial_state,
                                                         dtype=tf.float32)
        return outputs,state


def _embedding_dropout(x,keep_prob,noise_shape=None, seed=None, name=None):
    """
    the implementation of lexicon-based dropout
    :param x: inputs, size:[batch_size,max_seq_len,embedding_size]
    :param keep_prob: keep probabilities, size:[batch_size,max_seq_len]
    :param noise_shape: default like the size of keep_prob
    :param seed:
    :param name:
    :return: a tensor of the same shape of 'x'
    """
    noise_shape = keep_prob.get_shape()
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape,seed=seed,dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    # binary tensor: [batch_size,max_seq_len]-->[batch_size,max_seq_len,embedding_size]
    binary_tensor = expand_last_dims(binary_tensor,tf.shape(x)[-1])
    ret = x * binary_tensor

    return ret

def embedding_dropout(x,keep_prob,is_train):
    keep_prob1 = expand_last_dims(keep_prob,tf.shape(x)[-1])
    # if is_train:
    #     x = _embedding_dropout(x,keep_prob)
    # else:
    #     x = keep_prob1*x
    x = x*keep_prob1
    return x

def expand_last_dims(inputs,size):
    # inputs:  shape  [a,a,b]-->[a,a,b,size]
    r = tf.rank(inputs)
    inputs = tf.expand_dims(inputs, axis=r)
    inputs = tf.concat(values = [inputs for _ in range(300)], axis=r)
    return inputs