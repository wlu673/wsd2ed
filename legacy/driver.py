from collections import OrderedDict
import numpy as np
from model import MainModel


def main():
    embeddings = {'word': np.array([[0., 0., 0.],
                                    [0.4, 0.5, 0.6],
                                    [-0.1, -0.2, -0.3],
                                    [0.1, -0.4, 0.8]], dtype='float32'),
                  'anchor': np.array([[0., 0., 0., 0.],
                                      [0.1, 0.1, -0.3, 0.2],
                                      [-0.3, 0.2, 0.2, -0.3],
                                      [-0.1, -0.6, 0.6, -0.3],
                                      [-0.3, 0.2, 0.1, -0.3],
                                      [0.1, 0.0, -0.3, 0.2]], dtype='float32')}

    params = {'embeddings': embeddings,
              'update_embs': True,
              'features': OrderedDict([('word', 0), ('anchor', 0)]),
              'features_dim': OrderedDict([('word', 3), ('anchor', 4)]),
              'use_bin_fea': False,
              'len_contexts': 5,
              'num_category': 10,
              'wed_window': 2,
              'cnn_filter_num': 3,
              'cnn_filter_wins': [2],
              'cnn_multilayer_nn': [4],
              'batch_size': 3,
              'dropout': 0.0,
              'regularizer': 0.0,
              'lr': 0.01,
              'norm_lim': 0.0,
              'optimizer': 'adadelta',
              'kGivens': dict()}

    words = [[0, 1, 2, 3, 0],
             [1, 3, 2, 1, 1],
             [2, 2, 3, 2, 2]]

    anchor = [[0, 2, 3, 4, 0],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]]

    anchor_position = [2, 2, 2]

    candidate = [[2, 0, 5, 0, 0],
                 [3, 1, 2, 4, 0],
                 [4, 3, 6, 7, 8]]

    key = [0, 1, 2]

    M = MainModel(params)

    print '\nTraining ...\n'
    for i in range(1000):
        cost = M.f_grad_shared(words, anchor, anchor_position, candidate, key)
        M.f_update_param(params['lr'])
        for fea in M.container['embeddings']:
            M.container['set_zero'][fea](M.container['zero_vecs'][fea])
        print '>>> Epoch', i, ': cost = ', cost

    print '\nTesting ...\n'
    print M.f_pred(words, anchor, anchor_position, candidate)


if __name__ == '__main__':
    main()
