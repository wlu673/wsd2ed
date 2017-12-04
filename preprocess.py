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
    
    datNames = ['train', 'valid', 'sense02', 'sense03', 'sense07', 'eventTest', 'eventValid']
    maxLens = defaultdict(int)
    maxCandidate = -1
    maxNumFeature = -1
    wordCounter = defaultdict(int)
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
                
                for sen in els[4].split(';'): senseCounter[sen] += 1
                for sen in els[5].split(';'): senseCounter[sen] += 1
                
                if len(els[5].split(';')) > maxCandidate: maxCandidate = len(els[5].split(';'))
                
                if len(els[6].split()) > maxNumFeature: maxNumFeature = len(els[6].split())
                
                if 'sense' not in datName and datName != 'eventTest':
                    for fet in els[6].split(): fetCounter[fet] += 1
    
    print 'number of words in counter: ', len(wordCounter)
    print 'number of senses in counter: ', len(senseCounter)
    print 'number of features in counter: ', len(fetCounter)
    print '------maxLen----'
    for k in maxLens: print k, ' : ', maxLens[k]
    print 'maximum number of candidates: ', maxCandidate
    print 'maximum number of features: ', maxNumFeature
    
    return maxLens, wordCounter, senseCounter, fetCounter, maxCandidate, maxNumFeature

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
    embType = sys.argv[1]
    w2v_file = sys.argv[2]
    iFolder = sys.argv[3]
    oFolder = sys.argv[4]
    
    fetFreq = 1
    if len(sys.argv) >= 6: fetFreq = int(sys.argv[5])
    
    maxLens, wordCounter, senseCounter, fetCounter, maxCandidate, maxNumFeature = readInputFiles(iFolder)
    
    print "loading word embeddings...",
    dimEmb = 300
    if embType == 'word2vec':
        dimEmb, w2v = load_bin_vec(w2v_file, wordCounter)
    else:
        dimEmb, w2v = load_text_vec(w2v_file, wordCounter)
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
    dictionaries['senseId'] = makeDict(senseCounter)
    dictionaries['featureId'] = makeDict(fetCounter, False, fetFreq)
    print 'number of features: ', len(dictionaries['featureId'])
    print 'number of senses: ', len(dictionaries['senseId'])
    dictionaries['maxCandidate'] = maxCandidate
    dictionaries['maxNumFeature'] = maxNumFeature
    
    print 'dumping ...'
    cPickle.dump([embeddings, dictionaries], open(oFolder + '/' + embType + '.pkl', 'wb'))
    print "dataset created!"

if __name__=='__main__':
    main()
