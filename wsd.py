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
import theano.tensor.shared_randomstreams
from model import *
from data_iterator import *

##################################################################

scoreScript = {'wsd' : '/scratch/wl1191/wsd2ed2/scorers/scorer2',
               'event' : '/scratch/wl1191/wsd2ed2/scorers/eventScorer.py', }
               
paths_to_keys = {#'valid' : '/misc/kcgscratch1/ChoGroup/thien/projects/wsd/dataPreparer/sharingWeaver/datasetOneMil11-sampled/valid.key', #'/misc/kcgscratch1/ChoGroup/thien/projects/wsd/dataPreparer/sharingWeaver/dataset/valid.key',
                 'senseValid' : '/scratch/wl1191/wsd2ed2/data/Semcor/senseValid.key',
                 'sense02' : '/scratch/wl1191/wsd2ed2/data/Semcor/sense02.key',
                 'sense03' : '/scratch/wl1191/wsd2ed2/data/Semcor/sense03.key',
                 'sense07' : '/scratch/wl1191/wsd2ed2/data/Semcor/sense07.key',
                 'eventValid' : '/scratch/wl1191/wsd2ed2/data/Semcor/eventValid.key',
                 'eventTest' : '/scratch/wl1191/wsd2ed2/data/Semcor/eventTest.key', }

def prepareData(rev, embeddings, dictionaries, features, anchorMat, useBinaryFeatures):

    numAnchor = embeddings['anchor'].shape[0]-1
    
    batchSize = len(rev['word'])
    contextLen = len(rev['word'][0])
    
    anchorMat[::] = 0.
    
    for batchId in range(batchSize):
        for id in range(contextLen):
            anchor = numAnchor / 2 + id - rev['insAnchors'][batchId]
            if features['anchor'] == 0:
                scalar_anchor = 0
                if rev['word'][batchId][id] != 0:
                    scalar_anchor = anchor + 1
                anchorMat[batchId][id] = scalar_anchor
            elif features['anchor'] == 1:
                if rev['word'][batchId][id] != 0:
                    anchorMat[batchId][id][anchor] = 1.
    
    rev['anchor'] = anchorMat
    
    npdat = [ rev[fet] for fet in features if features[fet] >= 0 ]
    npdat += [ rev['insAnchors'] ]
    if useBinaryFeatures:
        npdat += [ rev['binaryFeatures'] ]
    npdat += [ rev['candidates'], rev['keys'] ]
        
    return npdat, rev

def predict(ncp, corpus, wsdModel, dictionaries, embeddings, features, anchorMat, useBinaryFeatures, idx2sense, outFile):

    wrdTypes, insIds, predictions = [], [], []
    for oneBatch, state in corpus:
        if not oneBatch['isValid']: break
        
        zippedCorpus, oneRev = prepareData(oneBatch, embeddings, dictionaries, features, anchorMat, useBinaryFeatures)

        if 'sense' in ncp:
            disams = wsdModel.pred_wsd(*zippedCorpus[0:-1])
        else:
            disams = wsdModel.pred_event(*zippedCorpus[0:-1])
        
        if state <= 0: state = len(disams)
        
        disams = [ idx2sense[d] for d in disams[0:state] ]
        
        predictions += disams
        insIds += oneRev['insIds'][0:state]
        wrdTypes += oneRev['wrdTypes'][0:state]

    assert len(predictions) == len(insIds)
    assert len(predictions) == len(wrdTypes)
    
    writer = open(outFile + '.unsorted', 'w')
    for wrd, iid, pred in zip(wrdTypes, insIds, predictions):
        writer.write(wrd + ' ' + iid + ' ' + pred + '\n')
    writer.close()
    
    sproc = subprocess.Popen(['sort', outFile + '.unsorted'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    sous, _ = sproc.communicate()
    writer = open(outFile, 'w')
    for line in sous.split('\n'):
        line = line.strip()
        if line: writer.write(line + '\n')
    writer.close()
    
    return score(outFile, paths_to_keys[ncp], ncp)

def score(predFile, keyFile, ncp):

    if 'event' in ncp:
        proc = subprocess.Popen(['python', scoreScript['event'], predFile, keyFile], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    else:
        proc = subprocess.Popen([scoreScript['wsd'], predFile, keyFile], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ous, _ = proc.communicate()
    
    p, r, f = 0., 0., 0.
    for line in ous.split('\n'):
        line = line.strip()
        if line.startswith('precision:'):
            els = line.split()
            p = float(els[1]) * 100.
        if line.startswith('recall:'):
            els = line.split()
            r = float(els[1]) * 100.
    if (p + r) > 0:
        f = (2*p*r) / (p+r)
    
    return {'p' : p, 'r' : r, 'f1' : f}

def generateParameterFileName(model, expected_features, contextLength, nhidden, conv_feature_map, conv_win_feature_map):
    res = model
    res += '.cw-'
    for fe in expected_features: res += str(expected_features[fe])
    res += '.cl-' + str(contextLength)
    res += '.h-' + str(nhidden)
    res += '.cf-' + str(conv_feature_map)
    res += '.cwf-'
    for wi in conv_win_feature_map: res += str(wi)
    return res

def train(dataset_path='',
          embedding_path='',
          model='basic',
          wedWindow=-1,
          expected_features = OrderedDict([('anchor', -1)]),
          contextLength=31,
          givenPath=None,
          updateEmbs=True,
          optimizer='adadelta',
          lr=0.01,
          dropout=0.05,
          regularizer=0.5,
          norm_lim = -1.0,
          verbose=1,
          decay=False,
          batch=50,
          multilayerNN1=[1200, 600],
          multilayerNN2=[1200, 600],
          nhidden=100,
          conv_feature_map=100,
          conv_win_feature_map=[2,3,4,5],
          lamb=0.01,
          seed=3435,
          nepochs=50,
          folder='./res',
          _params=None):
    
    os.environ["LC_ALL"] = "C"
    
    #preparing transfer knowledge
    kGivens = {}
    if givenPath and os.path.exists(givenPath):
        print '****Loading given knowledge in: ', givenPath
        kGivens = cPickle.load(open(givenPath, 'rb'))
    else: print givenPath, ' not exist'
    
    folder = '/scratch/wl1191/wsd2ed2/out/' + folder

    paramFolder = folder + '/params'

    if not os.path.exists(folder): os.mkdir(folder)
    if not os.path.exists(paramFolder): os.mkdir(paramFolder)
    
    paramFileName = paramFolder + '/' + generateParameterFileName(model, expected_features, contextLength, nhidden, conv_feature_map, conv_win_feature_map)

    print 'loading embeddings and dictionaries: ', embedding_path, ' ...'
    embeddings, dictionaries = cPickle.load(open(embedding_path, 'rb'))
    
    idx2word  = dict((k,v) for v,k in dictionaries['word'].iteritems())
    idx2sense  = dict((k,v) for v,k in dictionaries['senseId'].iteritems())

    emb_dimension = embeddings['word'].shape[1]
    startAnchor = 0
    endAnchor = (2*contextLength-1) + startAnchor
    embeddings['anchor'] = numpy.concatenate([numpy.zeros((1, embeddings['anchor'].shape[1]), dtype='float32'), embeddings['anchor'][startAnchor:endAnchor]], axis=0)

    features = OrderedDict([('word', 0)])
    for ffin in expected_features:
        features[ffin] = expected_features[ffin]
        if expected_features[ffin] == 0:
            print 'using features: ', ffin, ' : embeddings'
        elif expected_features[ffin] == 1:
            print 'using features: ', ffin, ' : binary'
    
    vocsize = len(idx2word)

    print 'vocabsize = ', vocsize, ', word embeddings dim = ', emb_dimension
    
    features_dim = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features:
        if features[ffin] == 1:
            features_dim[ffin] = embeddings[ffin].shape[0]-1
        elif features[ffin] == 0:
            features_dim[ffin] = embeddings[ffin].shape[1]
            
    datasetNames = ['senseTrain', 'eventTrain', 'senseValid', 'sense02', 'sense03', 'sense07', 'eventValid', 'eventTest']
    datasets = {}
    for dn in datasetNames:
        datasets[dn] = TextIterator(dataset_path + '/' + dn + '.dat',
                                    dictionaries,
                                    batch_size=batch,
                                    maxLenContext=contextLength,
                                    toPredict= False if 'Train' in dn else True)
    
    del _params['dataset_path']
    del _params['embedding_path']
    del _params['givenPath']
    del _params['folder']
    params = {'model' : model,
              'wedWindow' : wedWindow,
              'kGivens' : kGivens,
              'nh' : nhidden,
              'ne' : vocsize,
              'batch' : batch,
              'embs' : embeddings,
              'dropout' : dropout,
              'regularizer': regularizer,
              'norm_lim' : norm_lim,
              'updateEmbs' : updateEmbs,
              'features' : features,
              'features_dim' : features_dim,
              'conv_winre' : contextLength,
              'numSenses' : len(dictionaries['senseId'])+1,
              'binaryFeatureDim' : len(dictionaries['featureId']),
              'word2idDict' : dictionaries['word'],
              'id2wordDict' : idx2word,
              'optimizer' : optimizer,
              'multilayerNN1' : multilayerNN1,
              'multilayerNN2' : multilayerNN2,
              'conv_feature_map' : conv_feature_map,
              'conv_win_feature_map' : conv_win_feature_map,
              'lamb' : lamb,
              '_params' : _params}
    
    # instanciate the model
    print 'building model ...'
    numpy.random.seed(seed)
    random.seed(seed)
    
    if model.startswith('#'):
        model = model[1:]
        params['model'] = model
        useBinaryFeatures = True
        params['useBinaryFeatures'] = useBinaryFeatures
        wsdModel = eval('hybridModel')(params)
    elif model.endswith('2'):
        model = model[:-1]
        params['model'] = model
        useBinaryFeatures = False
        params['useBinaryFeatures'] = useBinaryFeatures
        wsdModel = eval('twoNetsModel')(params)
    else:
        useBinaryFeatures = False
        params['useBinaryFeatures'] = useBinaryFeatures
        wsdModel = eval('mainModel')(params)
    print 'done'
    
    trainDataSense = datasets['senseTrain']
    trainDataEvent = datasets['eventTrain']
    evaluatingDataset = OrderedDict([
                                     ('senseValid', datasets['senseValid']),
                                     ('sense02', datasets['sense02']),
                                     ('sense03', datasets['sense03']),
                                     ('sense07', datasets['sense07']),
                                     ('eventValid', datasets['eventValid']),
                                     ('eventTest', datasets['eventTest']),
                                     ])
    
    _perfs = OrderedDict()
    
    anchorMat = numpy.ones((2,), dtype='int32')
    
    # training model
    best_f1 = -numpy.inf
    clr = lr
    s = OrderedDict()
    for e in xrange(nepochs):
        s['_ce'] = e
        tic = time.time()
        print '-------------------training in epoch: ', e, ' -------------------------------------'
        print '\nTraining on word sense disambiguation ...\n'
        miniId = -1
        for oneBatch, _ in trainDataSense:
            if not oneBatch['isValid'] : break
            
            if anchorMat.ndim == 1:
                anchorMat = numpy.zeros(oneBatch['word'].shape, dtype='int32') if features['anchor'] == 0 else numpy.zeros((oneBatch['word'].shape[0], oneBatch['word'].shape[1], embeddings['anchor'].shape[0]-1), dtype='int32')
            
            miniId += 1
            #if miniId >= 50: break
            zippedData, _ = prepareData(oneBatch, embeddings, dictionaries, features, anchorMat, useBinaryFeatures)
            
            wsdModel.f_grad_shared_sense(*zippedData)
            wsdModel.f_update_param_sense(clr)
            
            for ed in wsdModel.container['embDict']:
                wsdModel.container['setZero'][ed](wsdModel.container['zeroVecs'][ed])
                
            if verbose:
                if miniId % 50 == 0:
                    print 'epoch %i >> %2.2f'%(e,(miniId+1)),'completed in %.2f (sec) <<'%(time.time()-tic)
                    sys.stdout.flush()

        print '\nTraining on event detection ...\n'
        miniId = -1
        for oneBatch, _ in trainDataEvent:
            if not oneBatch['isValid']: break

            if anchorMat.ndim == 1:
                anchorMat = numpy.zeros(oneBatch['word'].shape, dtype='int32') if features[
                                                                                      'anchor'] == 0 else numpy.zeros(
                    (oneBatch['word'].shape[0], oneBatch['word'].shape[1], embeddings['anchor'].shape[0] - 1),
                    dtype='int32')

            miniId += 1
            # if miniId >= 50: break
            zippedData, _ = prepareData(oneBatch, embeddings, dictionaries, features, anchorMat,
                                        useBinaryFeatures)

            wsdModel.f_grad_shared_event(*zippedData)
            wsdModel.f_update_param_event(clr)

            for ed in wsdModel.container['embDict']:
                wsdModel.container['setZero'][ed](wsdModel.container['zeroVecs'][ed])

            if verbose:
                if miniId % 50 == 0:
                    print 'epoch %i >> %2.2f' % (e, (miniId + 1)), 'completed in %.2f (sec) <<' % (
                    time.time() - tic)
                    sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print 'evaluating in epoch: ', e

        for elu in evaluatingDataset:
            _perfs[elu] = predict(elu, evaluatingDataset[elu], wsdModel, dictionaries, embeddings, features, anchorMat, useBinaryFeatures, idx2sense, folder + '/' + elu + '.pred' + str(e))
        
        perPrint(_perfs)
        
        print 'saving parameters ...'
        wsdModel.save(paramFileName + '.i' + str(e) + '.pkl')
        
        #print 'saving output ...'
        #for elu in evaluatingDataset:
        #    saving(evaluatingDataset[elu], _predictions[elu], _probs[elu], idx2word, idx2label, idMappings[elu], folder + '/' + elu + str(e) + '.out')
        
        if _perfs['eventValid']['f1'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['eventValid']['f1']
            print '*************NEW BEST: epoch: ', e
            if verbose:
                perPrint(_perfs, len('Current Performance')*'-')

            for elu in _perfs:
                s[elu] = _perfs[elu]
            s['_be'] = e
            
            #subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            #subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        
        # learning rate decay if no improvement in 10 epochs
        if decay and abs(s['_be']-s['_ce']) >= 10: clr *= 0.5 
        if clr < 1e-5: break

    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'BEST RESULT: epoch: ', s['_be']
    perPrint(s, len('Current Performance')*'-')
    print ' with the model in ', folder

def perPrint(perfs, mess='Current Performance'):
    print '------------------------------%s-----------------------------'%mess
    for elu in perfs:
        if elu.startswith('_'): continue
        print '----', elu
        print 'Performance: ', str(perfs[elu]['p']) + '\t' + str(perfs[elu]['r']) + '\t' + str(perfs[elu]['f1'])
    
    print '------------------------------------------------------------------------------'

if __name__ == '__main__':
    pass
