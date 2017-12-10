import numpy as np

import cPickle as pkl
import gzip
import random
from random import shuffle

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    def __init__(self,
                 datasetName,
                 datasetFile,
                 dictionaries,
                 batch_size=50,
                 maxLenContext=31,
                 toPredict=False):
        self.name = datasetName
        self.dataset = fopen(datasetFile, 'r')
        self.wordDict = dictionaries['word']
        self.eventTypeDict = dictionaries['eventTypeId']
        self.senseDict = dictionaries['senseId']
        self.featureDict = dictionaries['featureId']
        self.maxCandidateEvent = dictionaries['maxCandidateEvent']
        self.maxCandidateSense = dictionaries['maxCandidateSense']
        self.maxNumFeature = dictionaries['maxNumFeature']

        self.batch_size = batch_size
        self.maxLenContext = maxLenContext

        self.toPredict = toPredict

        self.buffer_instanceIds = []
        self.buffer_wordTypes = []

        self.buffer_words = []

        self.buffer_anchors = []

        self.buffer_keys = []
        self.buffer_candidates = []

        self.buffer_features = []

        self.oneBat_buffer_instanceIds = []
        self.oneBat_buffer_wordTypes = []

        self.oneBat_buffer_words = []

        self.oneBat_buffer_anchors = []

        self.oneBat_buffer_keys = []
        self.oneBat_buffer_candidates = []

        self.oneBat_buffer_features = []

        self.k = batch_size * 100 #20

        self.insIds = [''] * batch_size
        self.wrdTypes = [''] * batch_size
        self.words = np.zeros((batch_size, self.maxLenContext), dtype='int32')
        self.insAnchors = np.zeros((batch_size,), dtype='int32')
        self.keys = np.zeros((batch_size,), dtype='int32')
        self.candidates = np.zeros((batch_size, self.maxCandidateSense + 1), dtype='int32')
        self.binaryFeatures = np.zeros((batch_size, self.maxNumFeature + 1), dtype='int32')

    def __iter__(self):
        return self

    def reset(self):
        self.dataset.seek(0)

        self.oneBat_buffer_instanceIds = []
        self.oneBat_buffer_wordTypes = []

        self.oneBat_buffer_words = []

        self.oneBat_buffer_anchors = []

        self.oneBat_buffer_keys = []
        self.oneBat_buffer_candidates = []

        self.oneBat_buffer_features = []

    def next(self):

        # fill buffers, if it's empty

        if len(self.buffer_words) < self.batch_size:

            while True:
                if len(self.buffer_words) >= self.k: break

                line = self.dataset.readline().strip()
                if not line: break

                els = line.split('\t')

                insId = els[0]
                wordType = els[1]
                words = els[2]
                anchor = int(els[3])
                skey = els[4]
                scands = els[5]
                features = els[6]

                if skey not in scands.split(';') and not self.toPredict: continue

                self.buffer_instanceIds.append(insId)
                self.buffer_wordTypes.append(wordType)
                self.buffer_words.append(words)
                self.buffer_anchors.append(anchor)
                self.buffer_keys.append(skey)
                self.buffer_candidates.append(scands)
                self.buffer_features.append(features)

            if not self.toPredict:
                seedRand = random.randint(0,999999)
                random.seed(seedRand)
                shuffle(self.buffer_instanceIds)
                random.seed(seedRand)
                shuffle(self.buffer_wordTypes)
                random.seed(seedRand)
                shuffle(self.buffer_words)
                random.seed(seedRand)
                shuffle(self.buffer_anchors)
                random.seed(seedRand)
                shuffle(self.buffer_keys)
                random.seed(seedRand)
                shuffle(self.buffer_candidates)
                random.seed(seedRand)
                shuffle(self.buffer_features)

            if len(self.oneBat_buffer_instanceIds) < self.batch_size:
                for ti in range(self.batch_size-len(self.oneBat_buffer_instanceIds)):
                    if ti >= len(self.buffer_instanceIds): break

                    self.oneBat_buffer_instanceIds.append(self.buffer_instanceIds[ti])
                    self.oneBat_buffer_wordTypes.append(self.buffer_wordTypes[ti])
                    self.oneBat_buffer_words.append(self.buffer_words[ti])
                    self.oneBat_buffer_anchors.append(self.buffer_anchors[ti])
                    self.oneBat_buffer_keys.append(self.buffer_keys[ti])
                    self.oneBat_buffer_candidates.append(self.buffer_candidates[ti])
                    self.oneBat_buffer_features.append(self.buffer_features[ti])

        state = -1

        if len(self.buffer_words) < self.batch_size:
            if len(self.buffer_words) > 0:
                state = len(self.buffer_words)

                for ti in range(self.batch_size - len(self.buffer_words)):
                    self.buffer_instanceIds.insert(0, self.oneBat_buffer_instanceIds[ti])
                    self.buffer_wordTypes.insert(0, self.oneBat_buffer_wordTypes[ti])
                    self.buffer_words.insert(0, self.oneBat_buffer_words[ti])
                    self.buffer_anchors.insert(0, self.oneBat_buffer_anchors[ti])
                    self.buffer_keys.insert(0, self.oneBat_buffer_keys[ti])
                    self.buffer_candidates.insert(0, self.oneBat_buffer_candidates[ti])
                    self.buffer_features.insert(0, self.oneBat_buffer_features[ti])
            else:
                self.reset()
                return {'isValid': None,
                        'insIds' : None,
                        'wrdTypes' : None,
                        'word' : None,
                        'insAnchors' : None,
                        'keys' : None,
                        'candidates' : None,
                        'binaryFeatures' : None}, -1

        self.words[::] = 0
        self.keys[:] = 0
        self.candidates[::] = 0
        self.binaryFeatures[::] = -1
        self.insAnchors[::] = 0
        for i in range(self.batch_size):

            self.insIds[i] = self.buffer_instanceIds.pop()
            self.wrdTypes[i] = self.buffer_wordTypes.pop()

            aa = self.buffer_anchors.pop()
            ws = self.buffer_words.pop().split()
            for wid in range(self.maxLenContext):
                if (wid+aa-self.maxLenContext/2) >= 0 and (wid+aa-self.maxLenContext/2) < len(ws):
                    self.words[i][wid] = self.wordDict[ws[wid+aa-self.maxLenContext/2]] if ws[wid+aa-self.maxLenContext/2] in self.wordDict else 1
            self.insAnchors[i] = self.maxLenContext/2

            if 'sense' in self.name:
                icands = self.buffer_candidates.pop().split(';')
                for ic in icands:
                    if ic not in self.senseDict:
                        print 'cannot find senseCandidate in dict: ', ic
                        exit()
                self.candidates[i][0] = len(icands)
                for icid in range(len(icands)):
                    self.candidates[i][1+icid] = self.senseDict[icands[icid]]

            ikeys = self.buffer_keys.pop().split(';')
            ikey = None
            if 'event' in self.name:
                for ik in ikeys:
                    if ik in self.eventTypeDict:
                        ikey = self.eventTypeDict[ik]
                        break
                if ikey is None:
                    print 'cannot find eventTypeKey in dict: ', ikeys
                    exit()
            else:
                for ik in ikeys:
                    if ik in self.senseDict:
                        ikey = self.senseDict[ik]
                        break
                if not ikey:
                    print 'cannot find senseKey in dict: ', ikeys
                    exit()

            iorder = -1
            if 'event' in self.name:
                iorder = ikey
            else:
                for iid, iod in enumerate(self.candidates[i][ 1:(1+self.candidates[i][0]) ]):
                    if ikey == iod:
                        iorder = iid
                        break
                if iorder == -1 and not self.toPredict:
                    print 'cannot find key in candidate list: '
                    exit()
            self.keys[i] = iorder

            sfets = self.buffer_features.pop().split()
            sfc = 0
            for sf in sfets:
                if sf in self.featureDict:
                    sfc += 1
                    self.binaryFeatures[i][sfc] = self.featureDict[sf]
            self.binaryFeatures[i][0] = sfc

        return {'isValid': 'OK',
                'insIds' : self.insIds,
                'wrdTypes' : self.wrdTypes,
                'word' : self.words,
                'insAnchors' : self.insAnchors,
                'keys' : self.keys,
                'candidates' : self.candidates,
                'binaryFeatures' : self.binaryFeatures}, state
