import numpy as np
import cPickle
from collections import defaultdict, OrderedDict
import sys, re
import random
#import urllib
#reload(sys)
#sys.setdefaultencoding('utf-8')

acceptSet = set(['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

# 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'

def lookup(mess, key, gdict, addOne):
    if key not in gdict:
        nk = len(gdict)
        if addOne: nk += 1
        gdict[key] = nk
        if mess:
            print mess, ': ', key, ' --> id = ', gdict[key]

def readInputFiles(iFolder):
    
    datNames = ['eventTrain', 'senseTrain', 'senseValid', 'sense02', 'sense03', 'sense07', 'eventTest', 'eventValid']
    maxLens = defaultdict(int)
    maxCandidateEvent = -1
    maxCandidateSense = -1
    maxNumFeature = -1
    wordCounter = defaultdict(int)
    eventCounter = defaultdict(int)
    senseCounter = defaultdict(int)
    fetCounter = defaultdict(int)
    
    # read data
    for datName in datNames:
        with open(iFolder + '/' + datName + '.dat') as f:
            for line in f:
                line = line.strip()
                els = line.split('\t')
                words = els[2].split()
                for w in words: wordCounter[w.lower()] += 1
                if maxLens['xWords'] < len(words): maxLens['xWords'] = len(words)

                if 'event' in datName:
                    for etype in els[4].split(';'):
                        eventCounter[etype] += 1
                    for etype in els[5].split(';'):
                        eventCounter[etype] += 1
                    if len(els[5].split(';')) > maxCandidateEvent:
                        maxCandidateEvent = len(els[5].split(';'))
                else:
                    for sense in els[4].split(';'):
                        senseCounter[sense] += 1
                    for sense in els[5].split(';'):
                        senseCounter[sense] += 1
                    if len(els[5].split(';')) > maxCandidateSense:
                        maxCandidateSense = len(els[5].split(';'))
                
                if len(els[6].split()) > maxNumFeature:
                    maxNumFeature = len(els[6].split())
                
                if 'sense' not in datName and datName != 'eventTest':
                    for fet in els[6].split(): fetCounter[fet] += 1
    
    print 'number of words in counter: ', len(wordCounter)
    print 'number of event types in counter: ', len(eventCounter)
    print 'number of senses in counter: ', len(senseCounter)
    print 'number of features in counter: ', len(fetCounter)
    print '------maxLen----'
    for k in maxLens: print k, ' : ', maxLens[k]
    print 'maximum number of event candidates: ', maxCandidateEvent
    print 'maximum number of sense candidates: ', maxCandidateSense
    print 'maximum number of features: ', maxNumFeature
    
    return maxLens, wordCounter, eventCounter, senseCounter, fetCounter, maxCandidateEvent, maxCandidateSense, maxNumFeature

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k))
    W[0] = np.zeros(k)
    W[1] = np.random.uniform(-0.25,0.25,k)
    word_idx_map[0] = '__OUT_OF_BOUNDARY_WORD__'
    word_idx_map[1] = '__UNKNOWN_WORD__'
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               dim = word_vecs[word].shape[0]
            else:
                f.read(binary_len)
    print 'dim: ', dim
    return dim, word_vecs
    
def load_text_vec(fname, vocab):
    word_vecs = {}
    count = 0
    dim = 0
    with open(fname, 'r') as f:
        for line in f:
            count += 1
            line = line.strip()
            if count == 1:
                if len(line.split()) < 10:
                    dim = int(line.split()[1])
                    print 'dim: ', dim
                    continue
                else:
                    dim = len(line.split()) - 1
                    print 'dim: ', dim
            word = line.split()[0]
            emStr = line[(line.find(' ')+1):]
            if word in vocab:
                word_vecs[word] = np.fromstring(emStr, dtype='float32', sep=' ')
                if word_vecs[word].shape[0] != dim:
                    print 'mismatch dimensions: ', dim, word_vecs[word].shape[0]
                    exit()
    print 'dim: ', dim
    return dim, word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def makeDict(counter, addOne=True, freq=1):
    res = {}
    for c in counter:
        if counter[c] >= freq:
            res[c] = len(res) + 1 if addOne else len(res)
    return res

def main():
    embType = 'text'
    w2v_file_bin = '/scratch/wl1191/wsd2ed/data/GoogleNews-vectors-negative300.bin'
    w2v_file_text = '/scratch/wl1191/wsd2ed/data/concatEmbeddings.txt'
    iFolder = '/scratch/wl1191/wsd2ed2/data/Semcor'
    oFolder = '/scratch/wl1191/wsd2ed2/data/Semcor_processed'
    fetFreq = 2

    maxLens, wordCounter, eventCounter, senseCounter, fetCounter, maxCandidateEvent, maxCandidateSense, maxNumFeature = readInputFiles(iFolder)
    
    print "loading word embeddings...",
    dimEmb = 300
    if embType == 'word2vec':
        dimEmb, w2v = load_bin_vec(w2v_file_bin, wordCounter)
    else:
        dimEmb, w2v = load_text_vec(w2v_file_text, wordCounter)
    print "word embeddings loaded!"
    print "num words already in word embeddings: " + str(len(w2v))
    
    #add_unknown_words(w2v, vocab, 1, dimEmb)
    W1, word_idx_map = get_W(w2v, dimEmb)
    
    dist_size = 1000
    dist_dim = 50
    D = np.random.uniform(-0.25,0.25,(dist_size,dist_dim))
    
    embeddings = {}
    embeddings['word'] = W1
    embeddings['anchor'] = D
    
    dictionaries = {}
    dictionaries['word'] = word_idx_map
    dictionaries['eventTypeId'] = makeDict(eventCounter)
    dictionaries['senseId'] = makeDict(senseCounter, False)
    dictionaries['featureId'] = makeDict(fetCounter, False, fetFreq)
    print 'number of features: ', len(dictionaries['featureId'])
    print 'number of eventTypes: ', len(dictionaries['eventTypeId'])
    print 'number of senses: ', len(dictionaries['senseId'])
    dictionaries['maxCandidateEvent'] = maxCandidateEvent
    dictionaries['maxCandidateSense'] = maxCandidateSense
    dictionaries['maxNumFeature'] = maxNumFeature
    
    print 'dumping ...'
    cPickle.dump([embeddings, dictionaries], open(oFolder + '/' + embType + '.fetFreq2.SemcorACE.NoShuffled.TwoNets.pkl', 'wb'))
    print "dataset created!"

if __name__=='__main__':
    main()
