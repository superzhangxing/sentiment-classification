# -*- coding: utf-8 -*-

import tensorflow as tf

class lstm(object):
    def __init__(self,num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="lstm"):
        self.num_layers = num_layers
        self.units = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope

        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else num_units
            unit = tf.contrib.rnn.LSTMCell(num_units)
            init = tf.tile(tf.Variable(tf.zeros([1, num_units])), [batch_size, 1])
            mask = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32), keep_prob=keep_prob,
                           is_train=is_train, mode='recurrent')

            self.inits.append(init)
            self.units.append(unit)
            self.dropout_mask.append(mask)

    def __call__(self,inputs,seq_len, keep_prob=1.0, is_train=None,cancat_layers = True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                unit = self.units[layer]
                init = self.inits[layer]
                mask = self.dropout_mask[layer]

                out,final_state = tf.nn.dynamic_rnn(unit,outputs[-1]*mask, seq_len, initial_state=init, dtype=tf.float32)

                outputs.append(out, axis = 2)

        return outputs[-1]


def dropout(args,keep_prob,is_train,mode='recurrent'):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)

        if mode == 'embedding':
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == 'recurrent' and len(args.get_shape().as_list()) == 3:
            noise_shape = args.get_shape()
        args = tf.cond(is_train, lambda:tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args
