# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):
        self.batch_size = 25
        self.embedding_size = 300
        self.hidden_size = 100
        self.max_sentence_len = 100
        self.learning_rate = 0.00001
        self.num_layers = 3
        self.epochs = 100
        self.num_classes = 2

