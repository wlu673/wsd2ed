import numpy
import time
import sys
import subprocess
import os
import random
import cPickle
import copy

import theano
from theano import tensor as T
from collections import OrderedDict, defaultdict
from theano.tensor.nnet import conv
#from theano.tensor.signal import downsample
from theano.tensor.signal import pool
import theano.tensor.shared_randomstreams

#########################SOME UTILITIES########################


def randomMatrix(r, c, scale=0.2):
    #W_bound = numpy.sqrt(6. / (r + c))
    W_bound = 1.
    return scale * numpy.random.uniform(low=-W_bound, high=W_bound,\
                   size=(r, c)).astype(theano.config.floatX)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def _slice(_x, n, dim):
    return _x[:,n*dim:(n+1)*dim]

###############################################################

##########################Optimization function################

def adadelta(ips,cost,names,parameters,gradients,lr,norm_lim,rho=0.95,eps=1e-6):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in zip(names, parameters)]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in zip(names, parameters)]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in zip(names, parameters)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, rho * rg2 + (1. - rho) * (g ** 2)) for rg2, g in zip(running_grads2, gradients)] 
    f_grad_shared = theano.function(ips, cost, updates=zgup+rg2up, on_unused_input='ignore')

    updir = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1. - rho) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]
    
    if norm_lim > 0:
        param_up = clipGradient(param_up, norm_lim, names)

    f_param_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def sgd(ips,cost,names,parameters,gradients,lr,norm_lim):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in zip(names, parameters)]
    gsup = [(gs, g) for gs, g in zip(gshared, gradients)]

    f_grad_shared = theano.function(ips, cost, updates=gsup, on_unused_input='ignore')

    pup = [(p, p - lr * g) for p, g in zip(parameters, gshared)]
    
    if norm_lim > 0:
        pup = clipGradient(pup, norm_lim, names)
    
    f_param_update = theano.function([lr], [], updates=pup, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def clipGradient(updates, norm, names):
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

###############################################################

def _dropout_from_layer(rng, layers, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    res = []
    for layer in layers:
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        res += [output]
    return res

###############################Models###############################

def getConcatenation(embDict, vars, features, features_dim, tranpose=False):
    xs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed] if not tranpose else vars[ed].T
            xs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            if not tranpose:
                xs += [vars[ed]]
            else:
                xs += [vars[ed].dimshuffle(1,0,2)]

    if len(xs) == 1:
        basex = xs[0]
    else:
        basex = T.cast(T.concatenate(xs, axis=2), dtype=theano.config.floatX)

    return basex

def getInverseConcatenation(embDict, vars, features, features_dim):
        
    ixs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed].T[::-1]
            ixs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            ixs += [vars[ed].dimshuffle(1,0,2)[::-1]]                

    if len(ixs) == 1:
        ibasex = ixs[0]
    else:
        ibasex = T.cast(T.concatenate(ixs, axis=2), dtype=theano.config.floatX)
    
    return ibasex

def createMatrix(random, kGivens, name, word2idDict=None, id2wordDict=None):
    if 'sofmaxMainModel_' in name: return random
    if name in kGivens:
        if kGivens[name].shape == random.shape:
            print '<------ Using given ', name, '------>'
            if name != 'word': return kGivens[name]
            gWord2idDict = kGivens['storingDicts']['word2idDict']
            gId2wordDict = kGivens['storingDicts']['id2wordDict']
            for id in range(random.shape[0]):
                if id <= 1:
                    random[id] = kGivens[name][id]
                    continue
                word = id2wordDict[id]
                if word not in gWord2idDict:
                    print 'cannot find word in given word2idDict: ', word
                    exit()
                gid = gWord2idDict[word]
                random[id] = kGivens[name][gid]
            return random
        else: print '<------', name, ' is a given knowledge but mismatch dimension ', kGivens[name].shape, random.shape, '------>'
    return random

def rnn_ff(inps, dim, hidden, batSize, prefix, params, names, kGivens={}):
    Wx  = theano.shared( createMatrix(randomMatrix(dim, hidden), kGivens, prefix + '_Wx') )
    Wh  = theano.shared( createMatrix(randomMatrix(hidden, hidden), kGivens, prefix + '_Wh') )
    bh  = theano.shared( createMatrix(numpy.zeros(hidden, dtype=theano.config.floatX), kGivens, prefix + '_bh') )
    #model.container['bi_h0']  = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wx, Wh, bh ] #, model.container['bi_h0']
    names += [ prefix + '_Wx', prefix + '_Wh', prefix + '_bh' ] #, 'bi_h0'

    def recurrence(x_t, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh)
        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def rnn_gru(inps, dim, hidden, batSize, prefix, params, names, kGivens={}):
    Wc = theano.shared( createMatrix(numpy.concatenate([randomMatrix(dim, hidden), randomMatrix(dim, hidden)], axis=1), kGivens, prefix + '_Wc') )

    bc = theano.shared( createMatrix(numpy.zeros(2 * hidden, dtype=theano.config.floatX), kGivens, prefix + '_bc') )

    U = theano.shared( createMatrix(numpy.concatenate([ortho_weight(hidden), ortho_weight(hidden)], axis=1), kGivens, prefix + '_U') )
    Wx = theano.shared( createMatrix(randomMatrix(dim, hidden), kGivens, prefix + '_Wx') )

    Ux = theano.shared( createMatrix(ortho_weight(hidden), kGivens, prefix + '_Ux') )

    bx = theano.shared( createMatrix(numpy.zeros(hidden, dtype=theano.config.floatX), kGivens, prefix + '_bx') )

    #model.container['bi_h0'] = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wc, bc, U, Wx, Ux, bx ] #, model.container['bi_h0']
    names += [ prefix + '_Wc', prefix + '_bc', prefix + '_U', prefix + '_Wx', prefix + '_Ux', prefix + '_bx' ] #, 'bi_h0'
    
    def recurrence(x_t, h_tm1):
        preact = T.dot(h_tm1, U)
        preact += T.dot(x_t, Wc) + bc

        r_t = T.nnet.sigmoid(_slice(preact, 0, hidden))
        u_t = T.nnet.sigmoid(_slice(preact, 1, hidden))

        preactx = T.dot(h_tm1, Ux)
        preactx = preactx * r_t
        preactx = preactx + T.dot(x_t, Wx) + bx

        h_t = T.tanh(preactx)

        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def ffBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names, kGivens={}):

    bi_h = rnn_ff(inps, dim, hidden, batSize, prefix, params, names, kGivens=kGivens)
    
    ibi_h = rnn_ff(iinps, dim, hidden, batSize, iprefix, params, names, kGivens=kGivens)

    _ibi_h = ibi_h[::-1]
    
    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2).dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep
    
def gruBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names, kGivens={}):

    bi_h = rnn_gru(inps, dim, hidden, batSize, prefix, params, names, kGivens=kGivens)
    
    ibi_h = rnn_gru(iinps, dim, hidden, batSize, iprefix, params, names, kGivens=kGivens)

    _ibi_h = ibi_h[::-1]

    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2).dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep

def ffForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    ix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    
    i_h = rnn_ff(ix, dimIn, hidden, batch, prefix, params, names, kGivens=kGivens)
    
    rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names, kGivens={}):
    iix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    ii_h = rnn_ff(iix, dimIn, hidden, batch, iprefix, params, names, kGivens=kGivens)
    
    _ii_h = ii_h[::-1]
    
    rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    return ffBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_ffbi', prefix + '_ffibi', params, names, kGivens=kGivens)
    
def gruForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    ix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    
    i_h = rnn_gru(ix, dimIn, hidden, batch, prefix, params, names, kGivens=kGivens)
    
    rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names, kGivens={}):
    iix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    ii_h = rnn_gru(iix, dimIn, hidden, batch, iprefix, params, names, kGivens=kGivens)
    
    _ii_h = ii_h[::-1]
    
    rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    return gruBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_grubi', prefix + '_gruibi', params, names, kGivens=kGivens)
    
###############################CONVOLUTIONAL CONTEXT####################################

def convolutionalLayer(inpu, feature_map, batch, length, window, dim, prefix, params, names, kGivens={}):
    down = window / 2
    up = window - down - 1
    zodown = T.zeros((batch, 1, down, dim), dtype=theano.config.floatX)
    zoup = T.zeros((batch, 1, up, dim), dtype=theano.config.floatX)
    
    inps = T.cast(T.concatenate([zoup, inpu, zodown], axis=2), dtype=theano.config.floatX)
    
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length + down + up, dim)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=filter_shape).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window)) )

    conv_b = theano.shared( createMatrix(numpy.zeros(filter_shape[0], dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_convL_W_' + str(window), prefix + '_convL_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

    conv_out = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))

    return conv_out.dimshuffle(0,2,1,3).flatten(3)
    
def convContextLs(inps, feature_map, convWins, batch, length, dim, prefix, params, names, kGivens={}):
    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = convolutionalLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_winL' + str(i), params, names, kGivens=kGivens)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=2), dtype=theano.config.floatX)

    return convRep

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
    
def LeNetConvPoolLayer(inps, feature_map, batch, length, window, dim, prefix, params, names, kGivens={}):
    fan_in = window * dim
    fan_out = feature_map * window * dim / (length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length, dim)
    pool_size = (length - window + 1, 1)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=filter_shape).astype(theano.config.floatX), kGivens, prefix + '_conv_W_' + str(window)) )

    conv_b = theano.shared( createMatrix(numpy.zeros(filter_shape[0], dtype=theano.config.floatX), kGivens, prefix + '_conv_b_' + str(window)) )

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_conv_W_' + str(window), prefix + '_conv_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

        
    conv_out_act = ReLU(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x')) #T.tanh
    #conv_output = downsample.max_pool_2d(input=conv_out_act, ds=pool_size, ignore_border=True)
    conv_output = pool.pool_2d(input=conv_out_act, ds=pool_size, ignore_border=True)

    return conv_output.flatten(2)

def convContext(inps, feature_map, convWins, batch, length, dim, prefix, params, names, kGivens={}):

    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = LeNetConvPoolLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_win' + str(i), params, names, kGivens=kGivens)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=1), dtype=theano.config.floatX)

    return convRep
    
def nonConsecutiveConvLayer2(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 2
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        
        return [_m1, _m2]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[1][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvLayer3(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 3
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2, i_m3):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        
        return [_m1, _m2, _m3]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[2][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvLayer4(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 4
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2, i_m3, i_m4):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        ati = i_m3 + T.dot(_x, Ws[3])
        _m4 = T.maximum(i_m4, ati)
        
        return [_m1, _m2, _m3, _m4]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[3][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvLayer5(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 5
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2, i_m3, i_m4, i_m5):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        ati = i_m3 + T.dot(_x, Ws[3])
        _m4 = T.maximum(i_m4, ati)
        ati = i_m4 + T.dot(_x, Ws[4])
        _m5 = T.maximum(i_m5, ati)
        
        return [_m1, _m2, _m3, _m4, _m5]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[4][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvNet(inps, feature_map, convWins, batch, length, dim, prefix, params, names, kGivens={}):

    cx = inps.dimshuffle(1,0,2)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = eval('nonConsecutiveConvLayer' + str(convWin))(cx, feature_map, batch, length, dim, prefix + '_nonCons_win' + str(i), params, names, kGivens=kGivens)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=1), dtype=theano.config.floatX)

    return convRep
    
#############################Multilayer NNs################################

def HiddenLayer(inputs, nin, nout, params, names, prefix, kGivens={}):
    W_bound = numpy.sqrt(6. / (nin + nout))
    multi_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(nin, nout)).astype(theano.config.floatX), kGivens, prefix + '_multi_W') )

    multi_b = theano.shared( createMatrix(numpy.zeros(nout, dtype=theano.config.floatX), kGivens, prefix + '_multi_b') )
    res = []
    for input in inputs:
        out = ReLU(T.dot(input, multi_W) + multi_b) # T.nnet.sigmoid
        res += [out]
    
    params += [multi_W, multi_b]
    names += [prefix + '_multi_W', prefix + '_multi_b']
    
    return res

def MultiHiddenLayers(inputs, hids, params, names, prefix, kGivens={}):
    
    hiddenVector = inputs
    id = 0
    for nin, nout in zip(hids, hids[1:]):
        id += 1
        hiddenVector = HiddenLayer(hiddenVector, nin, nout, params, names, prefix + '_layer' + str(id), kGivens=kGivens)
    return hiddenVector

#########################################################################################

class BaseModel(object):

    def __init__(self, args):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        #de :: dimension of the word embeddings
        cs :: word window context size
        '''
        self.container = {}
        
        self.args = args
        self.args['rng'] = numpy.random.RandomState(3435)
        self.args['dropout'] = args['dropout'] if args['dropout'] > 0. else 0.
        
        # parameters of the model
        
        self.container['params'], self.container['names'] = [], []
        
        self.container['embDict'] = OrderedDict()
        self.container['vars'] = OrderedDict()
        self.container['dimIn'] = 0

        print '******************FEATURES******************'
        for ed in self.args['features']:
            if self.args['features'][ed] == 0:
                if ed != 'word':
                    self.container['embDict'][ed] = theano.shared( createMatrix(self.args['embs'][ed].astype(theano.config.floatX), self.args['kGivens'], ed) )
                else:
                    self.container['embDict'][ed] = theano.shared( createMatrix(self.args['embs'][ed].astype(theano.config.floatX), self.args['kGivens'], ed, self.args['word2idDict'], self.args['id2wordDict']) )
                
                if self.args['updateEmbs']:
                    print '@@@@@@@ Will update embedding tables'
                    self.container['params'] += [self.container['embDict'][ed]]
                    self.container['names'] += [ed]

            if self.args['features'][ed] == 0:
                self.container['vars'][ed] = T.imatrix()
                dimAdding = self.args['embs'][ed].shape[1]
                self.container['dimIn'] += dimAdding        
            elif self.args['features'][ed] == 1:
                self.container['vars'][ed] = T.tensor3()
                dimAdding = self.args['features_dim'][ed]
                self.container['dimIn'] += dimAdding

            if self.args['features'][ed] >= 0:
                print 'represetation - ', ed, ' : ', dimAdding 

        print 'REPRESENTATION DIMENSION = ', self.container['dimIn']

        self.container['keys'] = T.ivector('keys') # label
        self.container['candidates'] = T.imatrix('candidates') # label
        self.container['lr'] = T.scalar('lr')
        self.container['anchor'] = T.ivector('anchorPosition')
        self.container['binaryFeatures'] = T.imatrix('binaryFeatures')
        self.container['zeroVector'] = T.vector('zeroVector')
    
    def buildFunctions(self, fetre, fetre_dropout, dim_inter):
    
        softmaxW = theano.shared( createMatrix(randomMatrix(self.args['numSenses'], dim_inter), self.args['kGivens'], 'sofmaxMainModel_W') )
        softmaxb = theano.shared( createMatrix(numpy.zeros(self.args['numSenses'], dtype=theano.config.floatX), self.args['kGivens'], 'sofmaxMainModel_b') )
        
        self.container['params'] += [softmaxW, softmaxb]
        self.container['names'] += ['sofmaxMainModel_W', 'sofmaxMainModel_b']
        
        def recurTrain(_rep_dropout, _cands, _key, _fW, _fb):
            _W = _fW[_cands[1:(_cands[0]+1)]]
            _b = _fb[_cands[1:(_cands[0]+1)]]
            _p_y_given_x_dropout = T.nnet.softmax(T.dot(_W, _rep_dropout) + _b)[0]
            
            return _p_y_given_x_dropout[_key]
        
        sscores, _ = theano.scan( fn=recurTrain, 
                                  sequences=[fetre_dropout, self.container['candidates'], self.container['keys']],
                                  outputs_info=[None],
                                  non_sequences=[softmaxW, softmaxb],
                                  n_steps=fetre_dropout.shape[0])
                                  
        def recurTest(_rep, _cands, _fW, _fb):
            _W = _fW[_cands[1:(_cands[0]+1)]]
            _b = _fb[_cands[1:(_cands[0]+1)]]
            _p_y_given_x = T.nnet.softmax(T.dot((1.0 - self.args['dropout']) * _W, _rep) + _b)[0]
            _id = T.argmax(_p_y_given_x)
            
            return _cands[1+_id]
        
        spreds, _ = theano.scan( fn=recurTest, 
                                  sequences=[fetre, self.container['candidates']],
                                  outputs_info=[None],
                                  non_sequences=[softmaxW, softmaxb],
                                  n_steps=fetre.shape[0])
        
        nll = -T.mean(T.log(sscores))
        
        if self.args['regularizer'] > 0.:
            for pp, nn in zip(self.container['params'], self.container['names']):
                if 'multi' in nn:
                    nll += self.args['regularizer'] * (pp ** 2).sum()
        
        y_pred = spreds
        
        gradients = T.grad( nll, self.container['params'] )

        classifyInput = [ self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0 ]        
        classifyInput += [ self.container['anchor'] ]
        
        if self.args['useBinaryFeatures']:
            classifyInput += [ self.container['binaryFeatures'] ]
        
        classifyInput += [ self.container['candidates'] ]
        
        # theano functions
        self.disambiguate = theano.function(inputs=classifyInput, outputs=y_pred, on_unused_input='ignore')

        trainInput = classifyInput + [self.container['keys']]

        self.f_grad_shared, self.f_update_param = eval(self.args['optimizer'])(trainInput,nll,self.container['names'],self.container['params'],gradients,self.container['lr'],self.args['norm_lim'])
        
        self.container['setZero'] = OrderedDict()
        self.container['zeroVecs'] = OrderedDict()
        for ed in self.container['embDict']:
            self.container['zeroVecs'][ed] = numpy.zeros(self.args['embs'][ed].shape[1],dtype='float32')
            self.container['setZero'][ed] = theano.function([self.container['zeroVector']], updates=[(self.container['embDict'][ed], T.set_subtensor(self.container['embDict'][ed][0,:], self.container['zeroVector']))])

    def save(self, folder):
        storer = {}
        storer['storingDicts'] = {}
        for param, name in zip(self.container['params'], self.container['names']):
            storer[name] = param.get_value()
        #storer['binaryFeatureDict'] = self.args['binaryFeatureDict']
        #storer['window'] = self.args['window']
        storer['storingDicts']['word2idDict'] = self.args['word2idDict']
        storer['storingDicts']['id2wordDict'] = self.args['id2wordDict']
        sp = folder
        print 'saving parameters to: ', sp
        cPickle.dump(storer, open(sp, "wb"))
        #for param, name in zip(self.container['params'], self.container['names']):
        #    numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

def localWordEmbeddingsTrigger(model):
    
    wedWindow = model.args['wedWindow']
    
    extendedWords = model.container['vars']['word']
    wleft = T.zeros((extendedWords.shape[0], wedWindow), dtype='int32')
    wright = T.zeros((extendedWords.shape[0], wedWindow), dtype='int32')
    extendedWords = T.cast(T.concatenate([wleft, extendedWords, wright], axis=1), dtype='int32')
    
    def recurrence(words, pos, eembs):
        fet = words[pos:(pos+2*wedWindow+1)]
        fet = eembs[fet].flatten()
        return [fet]
    
    rep, _ = theano.scan(fn=recurrence, sequences=[extendedWords, model.container['anchor']], n_steps=extendedWords.shape[0], non_sequences=[model.container['embDict']['word']], outputs_info=[None])
    
    dim_rep = (2*wedWindow+1) * model.args['embs']['word'].shape[1]
    
    return rep, dim_rep

class mainModel(BaseModel):
    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        fetre, dim_inter = eval(self.args['model'])(self)
        
        if self.args['wedWindow'] > 0:
            rep, dim_rep = localWordEmbeddingsTrigger(self)
            fetre = T.concatenate([fetre, rep], axis=1)
            dim_inter += dim_rep
        
        fetre_dropout = _dropout_from_layer(self.args['rng'], [fetre], self.args['dropout'])
        fetre_dropout = fetre_dropout[0]
            
        hids = [dim_inter] + self.args['multilayerNN1']
        
        mul = MultiHiddenLayers([fetre, fetre_dropout], hids, self.container['params'], self.container['names'], 'multiMainModel', kGivens=self.args['kGivens'])
        
        fetre, fetre_dropout = mul[0], mul[1]
        
        dim_inter = hids[len(hids)-1]
        
        self.buildFunctions(fetre, fetre_dropout, dim_inter)

class hybridModel(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        fModel, dim_model = eval(self.args['model'])(self)
        
        if self.args['wedWindow'] > 0:
            rep, dim_rep = localWordEmbeddingsTrigger(self)
            fModel = T.concatenate([fModel, rep], axis=1)
            dim_model += dim_rep
        
        fModel_dropout = _dropout_from_layer(self.args['rng'], [fModel], self.args['dropout'])
        fModel_dropout = fModel_dropout[0]
        
        nnhids = [dim_model] + self.args['multilayerNN2']
        
        nnmul = MultiHiddenLayers([fModel, fModel_dropout], nnhids, self.container['params'], self.container['names'], 'multiHybridModelNN', kGivens=self.args['kGivens'])
        
        fModel, fModel_dropout = nnmul[0], nnmul[1]
        
        dim_model = nnhids[len(nnhids)-1]
        
        #-----multilayer nn
        
        hids = [self.args['binaryFeatureDim']] + self.args['multilayerNN1']
        
        layer0_multi_W = theano.shared( createMatrix(randomMatrix(self.args['binaryFeatureDim'], hids[1]), self.args['kGivens'], 'l0_multiHybridModelBin_fW') )
        layer0_multi_b = theano.shared( createMatrix(numpy.zeros(hids[1], dtype=theano.config.floatX), self.args['kGivens'], 'l0_multiHybridModelBin_fb') )
        
        self.container['params'] += [layer0_multi_W, layer0_multi_b]
        self.container['names'] += ['l0_multiHybridModelBin_fW', 'l0_multiHybridModelBin_fb']
        
        def recurrence(bfi, Wmat, bvec):
            idx = bfi[1:(bfi[0]+1)]
            weights = T.sum(Wmat[idx], axis=0) + bvec
            return weights
        
        firstMapped, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures'], outputs_info=[None], non_sequences=[layer0_multi_W, layer0_multi_b], n_steps=self.container['binaryFeatures'].shape[0])
        
        firstMapped = T.nnet.sigmoid(firstMapped)
        
        if len(hids) == 2:
            fMulti = firstMapped
            fMulti_dropout = firstMapped
        else:
            hids = hids[1:]
            fetreArr = MultiHiddenLayers([firstMapped], hids, self.container['params'], self.container['names'], 'multiHybridModelBin', kGivens=self.args['kGivens'])
            fMulti = fetreArr[0]
            fMulti_dropout = fMulti
        dim_multi = hids[len(hids)-1]
        
        fetre = T.concatenate([fModel, fMulti], axis=1)
        fetre_dropout = T.concatenate([fModel_dropout, fMulti_dropout], axis=1)
        dim_inter = dim_model + dim_multi
            
        self.buildFunctions(fetre, fetre_dropout, dim_inter)

def alternateHead(model):

    dimIn = model.container['dimIn']
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateHeadC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    _ix = _x[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateHeadR', '_ab_ialternateHeadR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    return rnnHeadIn(model, _x, 2)
    
def alternateHeadForward(model):

    dimIn = model.container['dimIn']
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateHeadForwardC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateHeadForwardR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rnnHeadIn(model, _x, 1)
    
def alternateHeadBackward(model):

    dimIn = model.container['dimIn']
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateHeadBackwardC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateHeadBackwardR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x[::-1].dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rnnHeadIn(model, _x, 1)

def alternateMax(model):

    dimIn = model.container['dimIn']
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateMaxC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    _ix = _x[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateMaxR', '_ab_ialternateMaxR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = 2 * model.args['nh']
    
    return fRnn, dim_rnn

def alternateMaxForward(model):

    dimIn = model.container['dimIn']
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateMaxForwardC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateMaxForwardR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = model.args['nh']
    
    return fRnn, dim_rnn
    
def alternateMaxBackward(model):

    dimIn = model.container['dimIn']
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateMaxBackwardC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateMaxBackwardR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x[::-1].dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = model.args['nh']
    
    return fRnn, dim_rnn
    
def alternateConv(model):

    _x = gruBiDirect(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'alternateConvR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    dimIn = 2 * model.args['nh']
    
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, '_ab_alternateConvC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])

    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def alternateConvForward(model):

    _x = gruForward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'alternateConvForwardR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    dimIn = model.args['nh']
    
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, '_ab_alternateConvForwardC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])

    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def alternateConvBackward(model):

    _x = gruBackward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'alternateConvBackwardR', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    dimIn = model.args['nh']
    
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, '_ab_alternateConvBackwardC', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])

    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
##
def convolute(model):
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
        
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], model.container['dimIn'], 'convolute', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
        
    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def nonConsecutiveConvolute(model):
    _x = getConcatenation(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], tranpose=False)
        
    fConv = nonConsecutiveConvNet(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], model.container['dimIn'], 'nonConsecutiveConvolute', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
        
    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def rnnHeadNonConsecutiveConv(model):
    rep_noncon, dim_noncon = nonConsecutiveConvolute(model)
    rep_rnn, dim_rnn = rnnHead(model)
    return T.concatenate([rep_noncon, rep_rnn], axis=1), dim_noncon + dim_rnn
##
def rnnHead(model):
    _x = gruBiDirect(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnHead', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 2)
##  
def rnnHeadForward(model):
    _x = gruForward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnHeadForward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1)
##
def rnnHeadBackward(model):
    _x = gruBackward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnHeadBackward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1)
##
def rnnHeadFf(model):
    _x = ffBiDirect(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnHeadFf', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 2)
##
def rnnHeadFfForward(model):
    _x = ffForward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnHeadFfForward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1)
##
def rnnHeadFfBackward(model):
    _x = ffBackward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnHeadFfBackward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1)
##
def rnnHeadIn(model, _x, num):
    
    def recurrence1(x_i, anchor):
        fet = x_i[anchor]
        return [fet]
    
    fRnn, _ = theano.scan(fn=recurrence1, sequences=[_x, model.container['anchor']], outputs_info=[None], n_steps=_x.shape[0])
        
    dim_rnn = num * model.args['nh']
    
    return fRnn, dim_rnn
##
def rnnMax(model):
    _x = gruBiDirect(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'rnnMax', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 2)
##    
def rnnMaxForward(model):
    _x = gruForward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'rnnMaxForward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)
##
def rnnMaxBackward(model):
    _x = gruBackward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'rnnMaxBackward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)
##  
def rnnMaxFf(model):
    _x = ffBiDirect(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'rnnMaxFf', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 2)
##
def rnnMaxFfForward(model):
    _x = ffForward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'rnnMaxFfForward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)
##
def rnnMaxFfBackward(model):
    _x = ffBackward(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'] , model.args['nh'], model.args['batch'], 'rnnMaxFfBackward', model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)

##
def rnnMaxIn(model, _x, num):
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = num * model.args['nh']
    
    return fRnn, dim_rnn

######################
def rnnAtt(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnAtt', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    IW = theano.shared(randomMatrix(2 * model.args['nh' + str(i)], 1))
    Ib = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        
    model.container['params'] += [IW, Ib]
    model.container['names'] += ['rnnAt_IW', 'rnnAT_Ib']     
        
    def recurrence(x_i):
        alpha = T.dot(x_i, IW) + Ib
        alpha = T.exp(alpha)
        alpha = alpha / T.sum(alpha)
        fet = (x_i * T.addbroadcast(alpha, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
        
    fRnn, _ = theano.scan(fn=recurrence, \
            sequences=_x, outputs_info=[None], n_steps=_x.shape[0])
                
    dim_rnn = 2 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def rnnAttHead(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnAtt', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    IW = theano.shared(randomMatrix(2 * model.args['nh' + str(i)], 1))
    Ib = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        
    model.container['params'] += [IW, Ib]
    model.container['names'] += ['rnnAt_IW', 'rnnAT_Ib']     
        
    def recurrenceAtt(x_i):
        alpha = T.dot(x_i, IW) + Ib
        alpha = T.exp(alpha)
        alpha = alpha / T.sum(alpha)
        fet = (x_i * T.addbroadcast(alpha, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
        
    fRnnAtt, _ = theano.scan(fn=recurrenceAtt, \
            sequences=_x, outputs_info=[None], n_steps=_x.shape[0])
            
    def recurrenceHead(x_i, pos1, pos2):
        fet = T.cast(T.concatenate([x_i[pos1], x_i[pos2]]), dtype=theano.config.floatX)
        return [fet]
        
    fRnnHead, _ = theano.scan(fn=recurrenceHead, \
            sequences=[_x, model.container['pos1' + str(i)], model.container['pos2' + str(i)]], outputs_info=[None], n_steps=_x.shape[0])
    
    fRnn = T.cast(T.concatenate([fRnnAtt, fRnnHead], axis=1), dtype=theano.config.floatX)
                
    dim_rnn = 6 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn

#####################################
