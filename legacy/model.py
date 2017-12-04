import theano, theano.tensor as T, theano.tensor.shared_randomstreams
from theano.tensor.nnet import conv
from theano.tensor.signal import pool
from collections import OrderedDict
import numpy as np
import cPickle


###########################################################################
# Optimization Functions

def adadelta(inputs, cost, names, parameters, gradients, lr, norm_lim, rho=0.95, eps=1e-6):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad' % k)
                    for k, p in zip(names, parameters)]
    running_up2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rup2' % k)
                   for k, p in zip(names, parameters)]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2' % k)
                      for k, p in zip(names, parameters)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, rho * rg2 + (1. - rho) * (g ** 2)) for rg2, g in zip(running_grads2, gradients)]
    f_grad_shared = theano.function(inputs, cost, updates=zgup + rg2up, on_unused_input='ignore')

    updir = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1. - rho) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]

    if norm_lim > 0:
        param_up = clip_gradient(param_up, norm_lim, names)

    f_update_param = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore')

    return f_grad_shared, f_update_param


def sgd(ips, cost, names, parameters, gradients, lr, norm_lim):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in zip(names, parameters)]
    gsup = [(gs, g) for gs, g in zip(gshared, gradients)]

    f_grad_shared = theano.function(ips, cost, updates=gsup, on_unused_input='ignore')

    pup = [(p, p - lr * g) for p, g in zip(parameters, gshared)]

    if norm_lim > 0:
        pup = clip_gradient(pup, norm_lim, names)

    f_param_update = theano.function([lr], [], updates=pup, on_unused_input='ignore')

    return f_grad_shared, f_param_update


def clip_gradient(updates, norm, names):
    id = -1
    res = []
    for p, g in updates:
        id += 1
        if not names[id].startswith('word') and 'multi' not in names[id] and p.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(g), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm))
            scale = desired_norms / (1e-7 + col_norms)
            g = g * scale

        res += [(p, g)]
    return res


###########################################################################
# Nonconsecutive CNN

def cnn(model):
    X = get_concatenation(model.container['embeddings'],
                          model.container['vars'],
                          model.args['features'],
                          dim_num=3,
                          transpose=False)
    X = T.cast(X.reshape((X.shape[0], 1, X.shape[1], X.shape[2])), dtype=theano.config.floatX)

    rep_cnn = []
    for i, filter_win in enumerate(model.args['cnn_filter_wins']):
        rep_win = cnn_layer(X,
                            model.args['cnn_filter_num'],
                            model.args['batch_size'],
                            model.args['len_contexts'],
                            filter_win,
                            model.container['fea_dim'],
                            'cnn_filter_win' + str(i),
                            model.container['params'],
                            model.container['names'],
                            model.args['kGivens'])
        rep_cnn += [rep_win]

    rep_cnn = T.cast(T.concatenate(rep_cnn, axis=1), dtype=theano.config.floatX)
    dim_cnn = model.args['cnn_filter_num'] * len(model.args['cnn_filter_wins'])
    return rep_cnn, dim_cnn


def get_concatenation(embeddings, variables, features, dim_num=3, transpose=False):
    reps = []

    for fea in features:
        if features[fea] == 0:
            v = variables[fea]
            if not transpose:
                reps += [embeddings[fea][v]]
            else:
                reps += [embeddings[fea][v.T]] if dim_num == 3 else [embeddings[fea][v].dimshuffle(1, 0)]
        elif features[fea] == 1:
            if not transpose:
                reps += [variables[fea]]
            else:
                reps += [variables[fea].dimshuffle(1, 0, 2)] if dim_num == 3 else [variables[fea].dimshuffle(1, 0)]

    if len(reps) == 1:
        X = reps[0]
    else:
        axis = 2 if dim_num == 3 else 1
        X = T.cast(T.concatenate(reps, axis=axis), dtype=theano.config.floatX)
    return X


def cnn_layer(X, filter_num, batch_size, len_contexts, filter_win, dim_in, prefix, params, names, kGivens):
    fan_in = filter_win * dim_in
    fan_out = filter_num * filter_win * dim_in / (len_contexts - filter_win + 1)
    W_bound = np.sqrt(6. / (fan_in + fan_out))

    filter_shape = (filter_num, 1, filter_win, dim_in)
    image_shape = (batch_size, 1, len_contexts, dim_in)
    pool_size = (len_contexts - filter_win + 1, 1)

    W = create_shared(np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape).astype(theano.config.floatX),
                      kGivens,
                      prefix + '_W' + str(filter_win))

    b = create_shared(np.zeros(filter_shape[0], dtype=theano.config.floatX),
                      kGivens,
                      prefix + '_b' + str(filter_win))

    params += [W, b]
    names += [prefix + '_W' + str(filter_win), prefix + '_b' + str(filter_win)]

    conv_out = conv.conv2d(input=X, filters=W, filter_shape=filter_shape, image_shape=image_shape)
    conv_out_act = T.maximum(0.0, conv_out + b.dimshuffle('x', 0, 'x', 'x'))  # ReLU
    rep_win = pool.pool_2d(input=conv_out_act, ds=pool_size, ignore_border=True)

    return rep_win.flatten(2)


###########################################################################
# Multi-Hidden Layer NN

def multi_hidden_layers(inputs, dim_hids, params, names, prefix, kGivens):
    hidden_vector = inputs
    index = 0
    for dim_in, dim_out in zip(dim_hids, dim_hids[1:]):
        index += 1
        hidden_vector = hidden_layer(hidden_vector, dim_in, dim_out, params, names, prefix + '_layer' + str(index), kGivens)
    return hidden_vector


def hidden_layer(inputs, dim_in, dim_out, params, names, prefix, kGivens):
    bound = np.sqrt(6. / (dim_in + dim_out))
    W = create_shared(np.random.uniform(low=-bound, high=bound, size=(dim_in, dim_out)).astype(theano.config.floatX),
                      kGivens,
                      prefix + '_W')
    b = create_shared(np.zeros(dim_out, dtype=theano.config.floatX), kGivens, prefix + '_b')
    res = []
    for x in inputs:
        out = T.nnet.sigmoid(T.dot(x, W) + b)
        res += [out]

    params += [W, b]
    names += [prefix + '_W', prefix + '_b']

    return res


###########################################################################
# RNN

def rnn_gru(inputs, dim_in, dim_hidden, mask, prefix, params, names, kGivens):
    Uc = create_shared(np.concatenate([ortho_weight(dim_hidden), ortho_weight(dim_hidden)], axis=1),
                      kGivens,
                      prefix + '_Uc')
    Wc = create_shared(np.concatenate([random_matrix(dim_in, dim_hidden), random_matrix(dim_in, dim_hidden)], axis=1),
                      kGivens,
                      prefix + '_Wc')
    bc = create_shared(np.zeros(2 * dim_hidden, dtype=theano.config.floatX), kGivens, prefix + '_bc')

    Ux = create_shared(ortho_weight(dim_hidden), kGivens, prefix + '_Ux')
    Wx = create_shared(random_matrix(dim_in, dim_hidden), kGivens, prefix + '_Wx')
    bx = create_shared(np.zeros(dim_hidden, dtype=theano.config.floatX), kGivens, prefix + '_bx')

    gru_params = [Wc, bc, Uc, Wx, Ux, bx]
    params += gru_params
    names += [prefix + '_Wc', prefix + '_bc', prefix + '_Uc', prefix + '_Wx', prefix + '_Ux', prefix + '_bx']

    def _slice(_x, n):
        return _x[n * dim_hidden:(n + 1) * dim_hidden]

    def recurrence(x_t, m, h_tm1):
        h_tm1 = m * h_tm1
        preact = T.nnet.sigmoid(T.dot(h_tm1, Uc) + T.dot(x_t, Wc) + bc)

        r_t = _slice(preact, 0)
        u_t = _slice(preact, 1)

        h_t = T.tanh(T.dot(h_tm1, Ux) * r_t + T.dot(x_t, Wx) + bx)
        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _ = theano.scan(fn=recurrence,
                       sequences=[inputs, mask],
                       outputs_info=T.alloc(0., dim_hidden),
                       n_steps=inputs.shape[0])

    return h, gru_params


def random_matrix(row, column, scale=0.2):
    # bound = np.sqrt(6. / (row + column))
    bound = 1.
    return scale * np.random.uniform(low=-bound, high=bound, size=(row, column)).astype(theano.config.floatX)


def ortho_weight(dim):
    W = np.random.randn(dim, dim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


###########################################################################
# Model Utilities


def create_shared(random, kGivens, name):
    if name in kGivens:
        if name == 'ffnn_b' or kGivens[name].shape == random.shape:
            print '>>> Using given', name
            return theano.shared(kGivens[name])
        else:
            print '>>> Dimension mismatch with given', name, ': Given:', kGivens[name].shape, ', Actual:', random.shape
    return theano.shared(random)


def trigger_contexts(model):
    wed_window = model.args['wed_window']
    extended_words = model.container['vars']['word']
    padding = T.zeros((extended_words.shape[0], wed_window), dtype='int32')
    extended_words = T.cast(T.concatenate([padding, extended_words, padding], axis=1), dtype='int32')

    def recurrence(words, position, emb):
        indices = words[position:(position + 2 * wed_window + 1)]
        rep = emb[indices].flatten()
        return [rep]

    rep_contexts, _ = theano.scan(fn=recurrence,
                                  sequences=[extended_words, model.container['anchor_position']],
                                  n_steps=extended_words.shape[0],
                                  non_sequences=[model.container['embeddings']['word']],
                                  outputs_info=[None])

    dim_contexts = (2 * wed_window + 1) * model.args['embeddings']['word'].shape[1]
    return rep_contexts, dim_contexts


def dropout_from_layer(rng, layers, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p = 1-p because 1's indicate keep and p is prob of dropping
    res = []
    for layer in layers:
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        # The cast is important because int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        res += [output]
    return res


###########################################################################


class MainModel(object):
    def __init__(self, args):
        self.args = args
        self.args['rng'] = np.random.RandomState(3435)
        self.args['dropout'] = args['dropout'] if args['dropout'] > 0. else 0.

        self.container = {}
        self.prepare_features()
        self.define_vars()

        rep_cnn, rep_cnn_dropout, dim_cnn = self.get_cnn_rep()
        probs, preds = self.get_prob(rep_cnn, rep_cnn_dropout, dim_cnn)
        self.f_pred, self.f_grad_shared, self.f_update_param = self.build_funcs(probs, preds)
        self.container['set_zero'] = OrderedDict()
        self.container['zero_vecs'] = OrderedDict()
        for ed in self.container['embeddings']:
            self.container['zero_vecs'][ed] = np.zeros(self.args['embeddings'][ed].shape[1]).astype(theano.config.floatX)
            self.container['set_zero'][ed] = \
                theano.function([self.container['zero_vector']],
                                updates=[(self.container['embeddings'][ed],
                                          T.set_subtensor(self.container['embeddings'][ed][0, :],
                                                          self.container['zero_vector']))])

    def save(self, path_out):
        storer = dict()
        for param, name in zip(self.container['params'], self.container['names']):
            storer[name] = param.get_value()
        cPickle.dump(storer, open(path_out, 'w'))

    def prepare_features(self, header_width=80):
        self.container['fea_dim'] = 0
        self.container['params'], self.container['names'] = [], []
        self.container['embeddings'], self.container['vars'] = OrderedDict(), OrderedDict()

        print 'Features'.center(header_width, '-')
        print 'Will update embeddings' if self.args['update_embs'] else 'Will not update embeddings'
        for fea in self.args['features']:
            if self.args['features'][fea] == 0:
                self.container['embeddings'][fea] = create_shared(
                    self.args['embeddings'][fea].astype(theano.config.floatX),
                    self.args['kGivens'],
                    fea)

                if self.args['update_embs']:
                    self.container['params'] += [self.container['embeddings'][fea]]
                    self.container['names'] += [fea]

            dim_added = self.args['features_dim'][fea]
            self.container['fea_dim'] += dim_added
            self.container['vars'][fea] = T.imatrix() if self.args['features'][fea] == 0 else T.tensor3()
            print 'Using feature \'%s\' with dimension %d' % (fea, dim_added)

        print 'Total feature dimension:', self.container['fea_dim']
        print '-' * header_width

    def define_vars(self):
        self.container['anchor_position'] = T.ivector('anchor_position')
        self.container['candidate'] = T.imatrix('candidate')
        self.container['key'] = T.ivector('key')
        self.container['lr'] = T.scalar('lr')
        self.container['zero_vector'] = T.vector('zero_vector')

    def get_cnn_rep(self):
        rep_inter, dim_inter = cnn(self)

        if self.args['wed_window'] > 0:
            rep_contexts, dim_contexts = trigger_contexts(self)
            rep_inter = T.concatenate([rep_inter, rep_contexts], axis=1)
            dim_inter += dim_contexts

        rep_inter_dropout = dropout_from_layer(self.args['rng'], [rep_inter], self.args['dropout'])[0]

        dim_hids = [dim_inter] + self.args['cnn_multilayer_nn']
        results = multi_hidden_layers([rep_inter, rep_inter_dropout],
                                      dim_hids,
                                      self.container['params'],
                                      self.container['names'],
                                      'cnn_multi_nn',
                                      self.args['kGivens'])
        rep_cnn, rep_cnn_dropout = results[0], results[1]
        dim_cnn = dim_hids[-1]

        return rep_cnn, rep_cnn_dropout, dim_cnn

    def get_prob(self, rep_cnn, rep_cnn_dropout, dim_cnn):
        W = create_shared(
            np.random.uniform(low=-.2, high=.2, size=(self.args['num_category'], dim_cnn)).astype(theano.config.floatX),
            self.args['kGivens'],
            'softmax_W')
        b = create_shared(np.zeros(self.args['num_category'], dtype=theano.config.floatX),
                          self.args['kGivens'],
                          'softmax_b')

        self.container['params'] += [W, b]
        self.container['names'] += ['softmax_W', 'softmax_b']

        def _step_train(_rep, _cand, _key, _W, _b):
            _W_cand = _W[_cand[1:(_cand[0] + 1)]]
            _b_cand = _b[_cand[1:(_cand[0] + 1)]]
            _prob = T.nnet.softmax(T.dot(_W_cand, _rep) + _b_cand)[0]
            return _prob[_key]

        probs, _ = theano.scan(fn=_step_train,
                               sequences=[rep_cnn_dropout, self.container['candidate'], self.container['key']],
                               outputs_info=[None],
                               non_sequences=[W, b],
                               n_steps=rep_cnn_dropout.shape[0])

        def _step_test(_rep, _cand, _W, _b):
            _W_cand = _W[_cand[1:(_cand[0] + 1)]]
            _b_cand = _b[_cand[1:(_cand[0] + 1)]]
            _prob = T.nnet.softmax(T.dot((1.0 - self.args['dropout']) * _W_cand, _rep) + _b_cand)[0]
            _pred = T.argmax(_prob)
            return _cand[1 + _pred]

        preds, _ = theano.scan(fn=_step_test,
                               sequences=[rep_cnn, self.container['candidate']],
                               outputs_info=[None],
                               non_sequences=[W, b],
                               n_steps=rep_cnn.shape[0])

        return probs, preds

    def build_funcs(self, probs, preds):
        inputs = [self.container['vars'][fea] for fea in self.args['features']]
        inputs += [self.container['anchor_position'], self.container['candidate']]
        f_pred = theano.function(inputs, preds, on_unused_input='ignore')

        cost = -T.mean(T.log(probs))
        if self.args['regularizer'] > 0.:
            for name, param in zip(self.container['params'], self.container['names']):
                if 'multi' in name:
                    cost += self.args['regularizer'] * (param ** 2).sum()
        gradients = T.grad(cost, self.container['params'])
        inputs += [self.container['key']]
        f_grad_shared, f_update_param = eval(self.args['optimizer'])(inputs,
                                                                     cost,
                                                                     self.container['names'],
                                                                     self.container['params'],
                                                                     gradients,
                                                                     self.container['lr'],
                                                                     self.args['norm_lim'])
        return f_pred, f_grad_shared, f_update_param
