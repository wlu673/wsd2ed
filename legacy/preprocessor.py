import cPickle
import numpy as np
from collections import defaultdict, OrderedDict


def read_inputs(path_in, data_names):
    maxes = {'len_sent': -1, 'num_cand': -1, 'num_bin_fea': -1}
    counters = {'word': defaultdict(int), 'category': defaultdict(int), 'bin_fea': defaultdict(int)}

    for name in data_names:
        print 'Reading %s.dat ...' % name
        with open(path_in + name + '.dat', 'r') as file_in:
            for line in file_in:
                entries = line.strip().split('\t')

                sent = entries[2].split()
                for word in sent:
                    counters['word'][word.lower()] += 1
                if maxes['len_sent'] < len(sent):
                    maxes['len_sent'] = len(sent)

                for cate in entries[4].split(';'):
                    counters['category'][cate] += 1

                candidates = entries[5].split(';')
                for cate in candidates:
                    counters['category'][cate] += 1
                if len(candidates) > maxes['num_cand']:
                    maxes['num_cand'] = len(candidates)

                if 'sense' not in name and name != 'eventTest':
                    for fea in entries[6].split():
                        counters['bin_fea'][fea] += 1
                if len(entries[6].split()) > maxes['num_bin_fea']:
                    maxes['num_bin_fea'] = len(entries[6].split())

    print
    for item in counters:
        print 'Number of %s counted: %d' % (item, len(counters[item]))
    for item, key in zip(['sentence length', 'number of candidates', 'number of binary features'],
                         ['len_sent', 'num_cand', 'num_bin_fea']):
        print 'Maximum %s: %d' % (item, maxes[key])
    print
    return maxes, counters


def create_word_embedding(path_w2v_bin, path_w2v_text, emb_type, vocab):
    print "Vocab size: " + str(len(vocab))
    print "Loading word embeddings..."
    if emb_type == 'word2vec':
        dim_word_vecs, word_vecs = load_bin_vec(path_w2v_bin, vocab)
    else:
        dim_word_vecs, word_vecs = load_text_vec(path_w2v_text, vocab)
    print "Word embeddings loaded!"
    print "Number of words already in word embeddings: " + str(len(word_vecs))
    W, map_word2index = get_W(word_vecs, dim_word_vecs)

    return W, map_word2index


def load_bin_vec(path_w2v, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(path_w2v, 'rb') as fin:
        header = fin.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = fin.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(fin.read(binary_len), dtype='float32')
                dim = word_vecs[word].shape[0]
            else:
                fin.read(binary_len)
    print 'Word embedding dim:', dim
    return dim, word_vecs


def load_text_vec(path_w2v, vocab):
    word_vecs = {}
    count = 0
    dim = 0
    with open(path_w2v, 'r') as fin:
        for line in fin:
            count += 1
            line = line.strip()
            if count == 1:
                if len(line.split()) < 10:
                    dim = int(line.split()[1])
                    print 'Word embedding dim:', dim
                    continue
                else:
                    dim = len(line.split()) - 1
                    print 'Word embedding dim:', dim
            word = line.split()[0]
            em_str = line[(line.find(' ') + 1):]
            if word in vocab:
                word_vecs[word] = np.fromstring(em_str, dtype='float32', sep=' ')
                if word_vecs[word].shape[0] != dim:
                    print 'Found a word with mismatched dimension:', dim, word_vecs[word].shape[0]
                    exit()
    print 'Word embedding dim:', dim
    return dim, word_vecs


def get_W(word_vecs, dim_word_vecs=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    map_word2idx = dict()

    W = np.zeros(shape=(vocab_size + 2, dim_word_vecs))
    W[1] = np.random.uniform(-0.25, 0.25, dim_word_vecs)
    map_word2idx[0] = '__OUT_OF_BOUNDARY_WORD__'
    map_word2idx[1] = '__UNKNOWN_WORD__'

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        map_word2idx[word] = i
        i += 1
    return W, map_word2idx


def create_anchor_embedding(len_contexts, dim_anchor_emb):
    D = np.random.uniform(-0.25, 0.25, (len_contexts + 1, dim_anchor_emb))
    D[0] = np.zeros(dim_anchor_emb)
    return D


def create_idx_map(counter, freq_threshold=1):
    map_item2idx = {}
    for item in counter:
        if counter[item] >= freq_threshold:
            map_item2idx[item] = len(map_item2idx)
    return map_item2idx


def make_data(path_in, path_out, data_names, len_contexts, map_item2idx, maxes):
    for name in data_names:
        print 'Reading %s.dat ...' % name
        with open(path_out + name + '.dat', 'w') as file_out:
            with open(path_in + name + '.dat', 'r') as file_in:
                for line in file_in:
                    entries = line.strip().split('\t')
                    file_out.write(entries[0] + '\t' + entries[1] + '\t')

                    process_sent(entries[2].split(), int(entries[3]), len_contexts, map_item2idx['word'], file_out)
                    if not process_labelNcategories(name == 'train',
                                                    entries[4].split(';')[0],
                                                    entries[5].split(';'),
                                                    map_item2idx['category'],
                                                    maxes['num_cand'],
                                                    file_out):
                        print 'Error: Cannot find label in candidates'
                        print 'Line:', line
                        exit(0)
                    process_bin_fea(entries[6].split(), map_item2idx['bin_fea'], maxes['num_bin_fea'], file_out)
                    file_out.write('\n')


def process_sent(sent, anchor_pos, len_contexts, map_word2idx, file_out):
    anchor_pos_new = len_contexts / 2
    sent_str = ''
    anchor_str = ''
    for pos_new in range(len_contexts):
        pos = pos_new + anchor_pos - anchor_pos_new / 2
        if 0 <= pos < len(sent):
            sent_str += str(map_word2idx[sent[pos]]) if sent[pos] in map_word2idx else '1'
            anchor_str += str(1 + pos_new)
        else:
            sent_str += '0'
            anchor_str += '0'
        sent_str += ';'
        anchor_str += ';'
    file_out.write(sent_str[:-1] + '\t' + str(anchor_pos_new) + '\t' + anchor_str[:-1] + '\t')


def process_labelNcategories(is_train, label, candidates, map_cate2idx, max_num_cand, file_out):
    idx_cand = [map_cate2idx[c] for c in candidates]
    key = -1
    try:
        key = idx_cand.index(map_cate2idx[label])
    except ValueError:
        if is_train:
            return False
    file_out.write(str(key) + '\t' + str(len(idx_cand)))
    candidates_new = [str(i) for i in idx_cand] + ['0'] * (max_num_cand - len(idx_cand))
    for c in candidates_new:
        file_out.write(';' + c)
    file_out.write('\t')
    return True


def process_bin_fea(bin_fea, map_fea2idx, max_num_bin_fea, file_out):
    idx_bin_fea = list()
    for fea in bin_fea:
        if fea in map_fea2idx:
            idx_bin_fea += [str(map_fea2idx[fea])]

    idx_bin_fea += ['0'] * (max_num_bin_fea - len(idx_bin_fea))
    file_out.write(str(len(idx_bin_fea)))
    for bf in idx_bin_fea:
        file_out.write(';' + bf)


def main(path_in='/scratch/wl1191/codes/data/sample/Semcor/',
         path_out='/scratch/wl1191/codes/data/sample/Semcor_processed/',
         path_w2v_bin='/scratch/wl1191/codes/data/GoogleNews-vectors-negative300.bin',
         path_w2v_text='/scratch/wl1191/codes/data/concatEmbeddings.txt',
         emb_type='text',
         len_contexts=21,
         dim_anchor_emb=50,
         fea_freq_threshold=2):
    # data_names = ['train', 'valid', 'sense02', 'sense03', 'sense07', 'eventTest', 'eventValid']
    data_names = ['train', 'valid']
    maxes, counters = read_inputs(path_in, data_names)

    W, map_word2index = create_word_embedding(path_w2v_bin, path_w2v_text, emb_type, counters['word'].keys())
    embeddings = dict({'word': W})
    embeddings['anchor'] = create_anchor_embedding(len_contexts, dim_anchor_emb)

    map_item2idx = dict({'word': map_word2index})
    map_item2idx['category'] = create_idx_map(counters['category'])
    map_item2idx['bin_fea'] = create_idx_map(counters['bin_fea'], freq_threshold=fea_freq_threshold)

    features_dim = OrderedDict([('word', W.shape[1]),
                                ('anchor', dim_anchor_emb),
                                ('bin_fea', len(map_item2idx['bin_fea']))])

    cPickle.dump([embeddings, map_item2idx, features_dim, maxes, len_contexts], open(path_out + 'data.pkl', 'w'))
    make_data(path_in, path_out, data_names, len_contexts, map_item2idx, maxes)


if __name__ == '__main__':
    main()
