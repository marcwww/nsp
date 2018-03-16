import codecs
import collections
import os
import pickle
from random import shuffle
import json
import config
import jpype

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

dg_predicates=[]

word_dict=diction()
term_dict=diction()


dg_set=set()
var_set=set()
date_set=set()
num_set=set()
entity_set=set()
relation_set=set()

act_dict=diction()
GO='GO'
BOS='<BOS>'

word_tokens=collections.defaultdict(list)
term_tokens=collections.defaultdict(list)
act_tokens=collections.defaultdict(list)

def load():
    with open(os.path.join(opts.data, 'dg_predicates'), 'r') as f:
        dg_predicates.extend(f.read().split('\n'))

    for fname in ('train.json','test.json'):
        path=os.path.join(opts.data,fname)
        with codecs.open(path,'r','utf-8') as f:
            f_js=json.load(f,encoding='utf-8')
            for sample in f_js:
                sen,lr_raw=sample['utterance'],sample['targetFormula']
                sen=sen.split(' ')
                word_dict.insert_batch(sen)
                word_tokens[fname].append(sen)

                lr=ast.MRL(lr_raw)
                lr.parse()
                terms, acts, dates, nums, variables, dgs, entitys, relations\
                    =lr.terms_and_acts(dg_predicates)

                acts=[GO]+acts
                terms=[BOS]+terms
                # These make sure 'GO' and '<BOS>' are indexed with 0
                #  in the respective dictionary

                act_dict.insert_batch(acts)
                term_dict.insert_batch(terms)
                for date in dates:
                    date_set.add(term_dict[date])
                for num in nums:
                    num_set.add(term_dict[num])
                for var in variables:
                    var_set.add(term_dict[var])
                for entity in entitys:
                    entity_set.add(term_dict[entity])
                for relation in relations:
                    relation_set.add(term_dict[relation])
                for term in terms:
                    if term in dg_predicates:
                        dg_set.add(term_dict[term])

                term_tokens[fname].append(terms)
                act_tokens[fname].append(acts)

def iter_data(word_tokens, tran_actions, tree_tokens, fname):
    '''iterate through the examples'''
    idx = range(len(word_tokens[fname]))

    #shuffle the ids for each epoch during training
    if fname == 'train.json':
        shuffle(idx)

    for i in idx:
        yield word_tokens[fname][i], tran_actions[fname][i], tree_tokens[fname][i]
