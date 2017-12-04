import random
from random import shuffle
import theano
import numpy as np


class TextIterator:
    def __init__(self, path_data, maxes, len_contexts=21, batch_size=50, to_predict=False):
        self.data = open(path_data, 'r')
        self.len_contexts = len_contexts
        self.max_num_cand = maxes['num_cand']
        self.max_num_bin_fea = maxes['num_bin_fea']

        self.batch_size = batch_size
        self.max_size = batch_size * 100  # 20
        self.to_predict = to_predict

        self.items = ['inst_id', 'type', 'word', 'anchor_pos', 'anchor', 'key', 'candidate', 'bin_fea']
        self.buffer, self.buffer_one_batch, self.inputs = dict(), dict(), dict()
        for item in self.items:
            self.buffer[item] = []
            self.buffer_one_batch[item] = []
            self.inputs[item] = None

    def initiate_inputs(self):
        self.inputs['inst_id'] = [''] * self.batch_size
        self.inputs['type'] = [''] * self.batch_size
        self.inputs['word'] = np.zeros((self.batch_size, self.len_contexts), dtype='int32')
        self.inputs['anchor_pos'] = np.zeros((self.batch_size,), dtype='int32')
        self.inputs['anchor'] = np.zeros((self.batch_size, self.len_contexts), dtype='int32')
        self.inputs['key'] = np.zeros((self.batch_size,), dtype='int32')
        self.inputs['candidate'] = np.zeros((self.batch_size, self.max_num_cand + 1), dtype='int32')
        self.inputs['bin_fea'] = np.zeros((self.batch_size, self.max_num_bin_fea + 1), dtype='int32')

    def __iter__(self):
        return self

    def reset(self):
        self.data.seek(0)
        for item in self.items:
            self.buffer_one_batch[item] = []

    def next(self):
        if len(self.buffer['inst_id']) < self.batch_size:
            self.fill_buffer()

        num_in_batch = -1
        if len(self.buffer['inst_id']) < self.batch_size:
            if len(self.buffer['inst_id']) > 0:
                num_in_batch = len(self.buffer['inst_id'])
                for i in range(self.batch_size - num_in_batch):
                    for item in self.items:
                        self.buffer[item].insert(0, self.buffer_one_batch[item][i])
            else:
                self.reset()
                return False, None, -1

        self.initiate_inputs()
        for index in range(self.batch_size):
            for item in ['inst_id', 'type']:
                self.inputs[item][index] = self.buffer[item].pop()
            for item in ['word', 'anchor', 'candidate', 'bin_fea']:
                self.inputs[item][index] = np.array([int(i) for i in self.buffer[item].pop().split(';')], dtype='int32')
            for item in ['anchor_pos', 'key']:
                self.inputs[item][index] = int(self.buffer[item].pop())
        self.inputs['bin_fea'].dtype = theano.config.floatX
        return True, self.inputs, num_in_batch

    def fill_buffer(self):
        while True:
            if len(self.buffer['inst_id']) >= self.max_size:
                break

            line = self.data.readline().strip()
            if not line:
                break

            entries = line.split('\t')
            for i, item in enumerate(self.items):
                self.buffer[item].append(entries[i])

        if not self.to_predict:
            seed = random.randint(0, 999999)
            for item in self.items:
                random.seed(seed)
                shuffle(self.buffer[item])

        if len(self.buffer_one_batch['inst_id']) < self.batch_size:
            for i in range(self.batch_size - len(self.buffer_one_batch['inst_id'])):
                if i >= len(self.buffer['inst_id']):
                    break
                for item in self.items:
                    self.buffer_one_batch[item].append(self.buffer[item][i])
