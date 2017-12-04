from wsd import train
from collections import OrderedDict
import sys
import argparse

def main(params):
    print params
    train(dataset_path = params['dataset_path'],
          embedding_path = params['embedding_path'],
          model = params['model'],
          wedWindow = params['wedWindow'],
          expected_features = params['expected_features'],
          contextLength = params['contextLength'],
          givenPath = params['givenPath'],
          updateEmbs = params['updateEmbs'],
          optimizer = params['optimizer'],
          lr = params['lr'],
          dropout = params['dropout'],
          regularizer = params['regularizer'],
          norm_lim = params['norm_lim'],
          verbose = params['verbose'],
          decay = params['decay'],
          batch = params['batch'],
          multilayerNN1 = params['multilayerNN1'],
          multilayerNN2 = params['multilayerNN2'],
          nhidden = params['nhidden'],
          conv_feature_map = params['conv_feature_map'],
          conv_win_feature_map = params['conv_win_feature_map'],
          seed = params['seed'],
          #emb_dimension=300, # dimension of word embedding
          nepochs = params['nepochs'],
          folder = params['folder'],
          _params = params)
def fetStr(ef):
    res = ''
    for f in ef:
        res += str(ef[f])
    return res

def fmStr(ft):
    res = ''
    for f in ft:
        res += str(f)
    return res

def argsp():
    aparser = argparse.ArgumentParser()
    
    aparser.add_argument('--dataset_path', help='path to the dataset file')
    aparser.add_argument('--embedding_path', help='path to the embedding and dictionary file')
    aparser.add_argument('--model', help='model to be used, see the code for potential options')
    aparser.add_argument('--wedWindow', help='window for local context (concatenation of surrouding emeddings)', type=int)
    aparser.add_argument('--contextLength', help='context window for input', type=int)    
    aparser.add_argument('--givenPath', help='path to the trained model parameters to initialize')
    aparser.add_argument('--updateEmbs', help='update embeddings during training or not')
    aparser.add_argument('--optimizer', help='optimier to use for training')
    aparser.add_argument('--lr', help='learning rate', type=float)
    aparser.add_argument('--dropout', help='dropout rate', type=float)
    aparser.add_argument('--regularizer', help='regularizer rate', type=float)
    aparser.add_argument('--norm_lim', help='normalization constant', type=float)
    aparser.add_argument('--verbose', help='print more info or not', type=int)
    aparser.add_argument('--decay', help='decay or not')
    aparser.add_argument('--batch', help='number of instances per batch', type=int)
    aparser.add_argument('--multilayerNN1', help='dimensions for the fist multiplayer neural nets', type=int, nargs='*')
    aparser.add_argument('--multilayerNN2', help='dimensions for the second multiplayer neural nets', type=int, nargs='*')
    aparser.add_argument('--nhidden', help='number of hidden units', type=int)
    aparser.add_argument('--conv_feature_map', help='number of filters for convolution', type=int)
    aparser.add_argument('--conv_win_feature_map', help='windows for filters for convolution', type=int, nargs='+')
    aparser.add_argument('--seed', help='random seed', type=int)
    aparser.add_argument('--nepochs', help='number of iterations to run', type=int)
    
    aparser.add_argument('--anchor', help='features : anchor', type=int)
    
    return aparser

if __name__=='__main__':
    
    pars={'dataset_path' : '/misc/kcgscratch1/ChoGroup/thien/projects/wsd/dataPreparer/event/data/Semcor',
          'embedding_path' : '/misc/kcgscratch1/ChoGroup/thien/projects/wsd/dataPreparer/event/word2vec.pkl.fetFreq2.SemcorACE.NoShuffled.pkl',
          'model' : 'convolute', # convolute # rnnHead, rnnMax, rnnHeadFf, rnnMaxFf, rnnHeadForward, rnnHeadBackward, rnnMaxForward, rnnMaxBackward, rnnHeadFfForward, rnnHeadFfBackward, rnnMaxFfForward, rnnMaxFfBackward # alternateHead, alternateMax, alternateConv, nonConsecutiveConvolute, rnnHeadNonConsecutiveConv
          'wedWindow' : 2,
          'expected_features' : OrderedDict([('anchor', 0),
                                            ]),
          'contextLength' : 21,
          'givenPath' : None,
          'updateEmbs' : True,
          'optimizer' : 'adadelta',
          'lr' : 0.01,
          'dropout' : 0.5,
          'regularizer' : 0.0,
          'norm_lim' : 9.0,
          'verbose' : 1,
          'decay' : False,
          'batch' : 50,
          'multilayerNN1' : [1200],
          'multilayerNN2' : [],
          'nhidden' : 300,
          'conv_feature_map' : 300,
          'conv_win_feature_map' : [2,3,4,5],
          'seed' : 3435,
          'nepochs' : 20,
          'folder' : './res'}
    
    args = vars(argsp().parse_args())
    for arg in args:
        if args[arg] != None:
            if arg == 'updateEmbs' or arg == 'decay':
                args[arg] = False if args[arg] == 'False' else True
            
            if arg == 'anchor':
                if args[arg] != 0 and args[arg] != 1: args[arg] = -1
                print '*****Updating feature parameters: ', arg, '(', pars['expected_features'][arg], ' --> ', args[arg], ')'
                pars['expected_features'][arg] = args[arg]
                continue          
            
            print '*****Updating default parameter: ', arg, '(', pars[arg], ' --> ', args[arg], ')'
            pars[arg] = args[arg]
    
    folder = 'model_' + pars['model'] \
             + '.h_' + str(pars['nhidden']) \
             + '.upd_' + str(pars['updateEmbs']) \
             + '.batch_' + str(pars['batch']) \
             + '.opt_' + pars['optimizer'] \
             + '.drop_' + str(pars['dropout']) \
             + '.reg_' + str(pars['regularizer']) \
             + '.fet_' + fetStr(pars['expected_features']) \
             + '.cl_' + str(pars['contextLength']) \
             + '.cvft_' + str(pars['conv_feature_map']) \
             + '.cvfm_' + fmStr(pars['conv_win_feature_map']) \
             + '.lr_' + str(pars['lr']) \
             + '.norm_' + str(pars['norm_lim']) \
             + '.s_' + str(pars['seed'])
    if pars['givenPath']: folder += '.gp'
    pars['folder'] =  'allReLu.sp.w21.newCand2.SemcorACE.NoShuffled.' + folder
    
    main(pars)
