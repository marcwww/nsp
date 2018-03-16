import codecs
import collections
import os
import pickle
from random import shuffle

import ast
from config import opts


class diction(object):
    def __init__(self):
        self._str2idx={}
        self._idx2str=[]

    def insert(self,str):
        # assert str not in self._str2idx
        if str in self._str2idx:
            return

        idx=len(self._str2idx)
        self._str2idx[str]=idx
        self._idx2str.append(str)

    def insert_batch(self,lst):
        for item in lst:
            self.insert(item)

    @property
    def size(self):
        return len(self._idx2str)

    @property
    def idx2str(self):
        return self._idx2str

    @property
    def str2idx(self):
        return self._str2idx

    def __getitem__(self, str):
        assert str in self._str2idx
        return self._str2idx[str]

    def save(self,fpath):
        with open(fpath,'wb') as f:
            pickle.dump((self._str2idx,self._idx2str),f,pickle.HIGHEST_PROTOCOL)

    def load(self,fpath):
        with open(fpath,'rb') as f:
            self._idx2str,self._idx2str=pickle.load(f)

gp_predicates=[]
word_dict=diction()
nt_dict=diction()
ter_dict=diction()
act_dict=diction()

word_tokens=collections.defaultdict(list)
term_tokens=collections.defaultdict(list)
act_tokens=collections.defaultdict(list)

def load():
    with open(os.path.join(opts.data, 'dg_predicates'), 'r') as f:
        gp_predicates.extend(f.read().split('\n'))

    for fname in ('train','valid','test'):
        path=os.path.join(opts.data,fname)
        with codecs.open(path,'r','utf-8') as f:
            for line in f:
                sen,lr_raw=line.rstrip().split('\t')
                sen=sen.split(' ')

                word_dict.insert_batch(sen)
                word_tokens[fname].append(sen)

                lr= ast.LogicRep(lr_raw)
                nlst = lr.parse()
                tree= ast.Tree(nlst)

                nlves,lves=tree.get_nlves_and_lves()
                ter_dict.insert_batch(lves)
                nt_dict.insert_batch(nlves)

                terms,actions=tree.get_terms_and_acts(gp_predicates)
                act_dict.insert_batch(actions)
                term_tokens[fname].append(terms)
                act_tokens[fname].append(actions)


def iter_data(word_tokens, tran_actions, tree_tokens, fname):
    '''iterate through the examples'''
    idx = range(len(word_tokens[fname]))

    #shuffle the ids for each epoch during training
    if fname == 'train':
        shuffle(idx)

    for i in idx:
        yield word_tokens[fname][i], tran_actions[fname][i], tree_tokens[fname][i]
