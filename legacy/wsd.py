import sys
import time
import random
import cPickle
import subprocess
import numpy as np
from collections import OrderedDict
from data_iterator import TextIterator
from model import MainModel


def train(model, data_train, features, lr, epoch):
    cost = 0
    print (' Training in epoch %d ' % epoch).center(80, '-')
    time_start = time.time()
    for is_valid, data_batch, _ in data_train:
        if not is_valid:
            break
        inputs = []
        for fea in features:
            inputs += [data_batch[fea]]
        inputs += [data_batch['anchor_pos'],
                   data_batch['candidate'],
                   data_batch['key']]
        cost += model.f_grad_shared(*inputs)
        model.f_update_param(lr)
        for fea in model.container['embeddings']:
            model.container['set_zero'][fea](model.container['zero_vecs'][fea])
    print 'Completed in %.2f seconds\nCost = %.5f' % (time.time() - time_start, cost)
    sys.stdout.flush()


def predict(model, data_eval, features, map_idx2cate):
    inst_ids, types, predictions = [], [], []
    for is_valid, data_batch, num_in_batch in data_eval:
        if not is_valid:
            break
        inputs = []
        for fea in features:
            inputs += [data_batch[fea]]
        inputs += [data_batch['anchor_pos'],
                   data_batch['candidate']]
        preds = model.f_pred(*inputs)
        if num_in_batch < 0:
            num_in_batch = len(preds)
        inst_ids += data_batch['inst_id'][0:num_in_batch]
        types += data_batch['type'][0:num_in_batch]
        predictions += [map_idx2cate[idx] for idx in preds[0:num_in_batch]]
    assert len(predictions) == len(inst_ids)
    assert len(predictions) == len(types)
    return inst_ids, types, predictions


def write_out(file_name, inst_ids, types, predictions):
    with open(file_name + '.unsorted', 'w') as file_out:
        for t, i, p in zip(types, inst_ids, predictions):
            file_out.write(t + ' ' + i + ' ' + p + '\n')
    proc = subprocess.Popen(['sort', file_name + '.unsorted'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    outputs, _ = proc.communicate()
    with open(file_name, 'w') as file_out:
        for line in outputs.split('\n'):
            line = line.strip()
            if line:
                file_out.write(line + '\n')


def score(path_pred, path_key, data_name, path_scorer_event, path_scorer_wsd):
    if 'event' in data_name:
        proc = subprocess.Popen(['python', path_scorer_event, path_pred, path_key],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
    else:
        proc = subprocess.Popen([path_scorer_wsd, path_pred, path_key], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    outputs, _ = proc.communicate()

    p, r, f = 0., 0., 0.
    for line in outputs.split('\n'):
        line = line.strip()
        if line.startswith('precision:'):
            p = float(line.split()[1]) * 100.
        if line.startswith('recall:'):
            r = float(line.split()[1]) * 100.
    if (p + r) > 0:
        f = (2 * p * r) / (p + r)

    return {'p': p, 'r': r, 'f1': f}


def print_perf(perf, msg='Current Performance'):
    print (' ' + msg + ' ').center(80, '-')
    for data_eval in perf:
        print '%s:\t%.2f\t%.2f\t%.2f' % (data_eval, perf[data_eval]['p'], perf[data_eval]['r'], perf[data_eval]['f1'])
    print '-' * 80


def main(path_in='/scratch/wl1191/codes/data/Semcor_processed/',
         path_out='/scratch/wl1191/codes/out/',
         path_key='/scratch/wl1191/codes/data/Semcor/',
         path_scorer_event='/scratch/wl1191/codes/scorers/eventScorer.py',
         path_scorer_wsd='/scratch/wl1191/codes/scorers/scorer2',
         path_kGivens='/scratch/wl1191/codes/out0-14/params14.pkl',
         update_embs=True,
         use_bin_fea=False,
         wed_window=2,
         cnn_filter_num=300,
         cnn_filter_wins=(2, 3, 4, 5),
         cnn_multilayer_nn=(1200,),
         batch_size=50,
         regularizer=0.0,
         lr=0.01,
         lr_decay=False,
         norm_lim=9.0,
         optimizer='adadelta',
         dropout=0.5,
         seed=3435,
         nepochs=20):

    # Prepare parameters
    embeddings, map_item2idx, features_dim, maxes, len_contexts = cPickle.load(open(path_in + 'data.pkl', 'r'))
    map_idx2cate = dict((v, k) for k, v in map_item2idx['category'].iteritems())

    features = OrderedDict([('word', 0), ('anchor', 0)])
    if use_bin_fea:
        features['bin_fea'] = 1

    kGivens = dict()
    if path_kGivens is not None:
        kGivens = cPickle.load(open(path_kGivens, 'r'))

    params = {'update_embs': update_embs,
              'features': features,
              'features_dim': features_dim,
              'use_bin_fea': use_bin_fea,
              'len_contexts': len_contexts,
              'num_category': len(map_item2idx['category']),
              'wed_window': wed_window,
              'cnn_filter_num': cnn_filter_num,
              'cnn_filter_wins': list(cnn_filter_wins),
              'cnn_multilayer_nn': list(cnn_multilayer_nn),
              'batch_size': batch_size,
              'dropout': dropout,
              'regularizer': regularizer,
              'lr': lr,
              'norm_lim': norm_lim,
              'optimizer': optimizer,
              'kGivens': kGivens}

    print 'Saving model configuration ...'
    cPickle.dump(params, open(path_out + 'model_config.pkl', 'w'))

    params['embeddings'] = embeddings

    # Prepare datasets
    datasets_names = ['train', 'valid', 'sense02', 'sense03', 'sense07', 'eventValid', 'eventTest']
    # datasets_names = ['train', 'valid']
    datasets = {}
    for dn in datasets_names:
        datasets[dn] = TextIterator(path_in + dn + '.dat', maxes, len_contexts, batch_size, dn != 'train')

    print 'Building model ...'
    np.random.seed(seed)
    random.seed(seed)
    model = MainModel(params)

    data_train = datasets['train']
    data_evaluate = OrderedDict([
        ('valid', datasets['valid']),
        ('sense02', datasets['sense02']),
        ('sense03', datasets['sense03']),
        ('sense07', datasets['sense07']),
        ('eventValid', datasets['eventValid']),
        ('eventTest', datasets['eventTest'])])

    perfs = OrderedDict()
    best_perf = OrderedDict()
    best_f1 = -np.inf
    best_epoch = -1
    curr_lr = lr
    sys.stdout.flush()
    for epoch in xrange(nepochs):
        train(model, data_train, features, params['lr'], epoch)
        for eval_name in data_evaluate:
            inst_ids, types, predictions = predict(model, data_evaluate[eval_name], features, map_idx2cate)
            file_name = path_out + eval_name + '.pred' + str(epoch)
            write_out(file_name, inst_ids, types, predictions)
            perfs[eval_name] = score(file_name, path_key + eval_name + '.key', eval_name, path_scorer_event, path_scorer_wsd)

        print '\n', 'Saving parameters'
        model.save(path_out + 'params' + str(epoch) + '.pkl')

        print print_perf(perfs)
        if perfs['valid']['f1'] > best_f1:
            best_f1 = perfs['valid']['f1']
            best_epoch = epoch
            for data_eval in perfs:
                best_perf[data_eval] = perfs[data_eval]
            print 'NEW BEST: Epoch', epoch

        # learning rate decay if no improvement in 10 epochs
        if lr_decay and abs(best_epoch - epoch) >= 10:
            curr_lr *= 0.5
        if curr_lr < 1e-5:
            break
        sys.stdout.flush()

    print '\n', '=' * 80, '\n'
    print 'BEST RESULT: Epoch', best_epoch
    print_perf(best_perf, 'Best Performance')


if __name__ == '__main__':
    main()
