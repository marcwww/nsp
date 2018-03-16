import os

import ast
import config
import data
import mdl
import jpype
import numpy as np


def query(uttr):
    nnparser = mdl.NNParser()

    if os.path.exists(config.opts.model):
        fname='epoch%03d.model' % (config.opts.epochs - 1)
        path=os.path.join(config.opts.model,fname)
        nnparser.load_mdl(path)

    sen=str(uttr).split(' ')
    acts,terms=nnparser.parse(sen)
    raw_lf = mdl.recover_rlf(acts, terms)
    return raw_lf

def get_sparql(output_actions, output_tokens):
    lDCS = mdl.recover_rlf(output_actions, output_tokens)
    return lDCS

def test(tag):
    nnparser = mdl.NNParser(pretrained_embedding=None)
    SparqlExecutor = jpype.JClass('edu.stanford.nlp.sempre.freebase.SparqlExecutor')

    if os.path.exists(config.opts.model):
        fname = 'epoch%03d.model' % (config.opts.epochs - 1)
        # fname = 'epoch%03d.model' % (40)
        path = os.path.join(config.opts.model, fname)
        nnparser.load_mdl(path)

    fname='test_res%s' % (tag)
    save_as=os.path.join(config.opts.result,fname)
    rf = open(save_as, 'w')
    test_sents = 0
    test_loss = 0.0
    i=0
    contrast_vec=[]
    for word_tokens, act_tokens, term_tokens in data. \
            iter_data(data.word_tokens, data.act_tokens,
                      data.term_tokens, 'test.json'):
        output_actions, output_tokens = \
            nnparser.parse(word_tokens)
        print i
        i+=1

        pre_lDCS=get_sparql(output_actions,output_tokens)
        gold_lDCS = get_sparql(act_tokens[1:], term_tokens[1:])
        contrast_vec.append(int(act_tokens[1:]==output_actions))
        # print act_tokens[1:],output_actions
        print 'Current accuracy:%f' % ((np.sum(contrast_vec)+0.0)/len(contrast_vec))

        rf.write(pre_lDCS+'\t'+gold_lDCS + '\n')

    rf.close()