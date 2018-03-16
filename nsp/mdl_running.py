import os

import ast
import config
import data
import mdl
import numpy as np

def query(uttr):
    nnparser = mdl.NNParser(word_dict=data.word_dict,
                            nt_dict=data.nt_dict,
                            ter_dict=data.ter_dict,
                            act_dict=data.act_dict,
                            dim_word=config.opts.dim_word,
                            dim_nt=config.opts.dim_nt,
                            dim_ter=config.opts.dim_ter,
                            dim_act=config.opts.dim_act,
                            dropout=config.opts.dropout,
                            dim_h=config.opts.dim_h,
                            nn_layers=config.opts.nn_layers)

    if os.path.exists(config.opts.model):
        fname='epoch%03d.model' % (config.opts.epochs - 1)
        path=os.path.join(config.opts.model,fname)
        nnparser.load_mdl(path)

    sen=str(uttr).split(' ')
    acts,terms=nnparser.parse(sen)
    raw_lf = mdl.recover_rlf(acts, terms)
    lf= ast.LogicRep(raw_lf)

    tokens=lf.reconvert()
    prolog_querry=''.join(tokens)

    # print raw_lf
    # print prolog_querry

    script_name='wasp-1.0/eval.pl'
    script_path=os.path.join(config.home,script_name)

    cmd = 'echo "execute_funql_query({}, X)." | swipl -s {} 2>&1  | grep "X ="'.\
        format(prolog_querry, script_path)

    return raw_lf,prolog_querry,os.popen(cmd).readlines()

def exec_prolog(prolog_query):
    script_name = 'wasp-1.0/eval.pl'
    script_path = os.path.join(config.home, script_name)

    cmd = 'echo "execute_funql_query({}, X)." | swipl -s {} 2>&1  | grep "X ="'. \
        format(prolog_query, script_path)

    return os.popen(cmd).readlines()

def get_ans(actions, tokens):
    raw_lf = mdl.recover_rlf(actions, tokens)
    lf = ast.LogicRep(raw_lf)
    tokens = lf.reconvert()
    prolog_query = ''.join(tokens)
    res = exec_prolog(prolog_query)
    return res,prolog_query

def test(tag):
    nnparser = mdl.NNParser(word_dict=data.word_dict,
                            nt_dict=data.nt_dict,
                            ter_dict=data.ter_dict,
                            act_dict=data.act_dict,
                            dim_word=config.opts.dim_word,
                            dim_nt=config.opts.dim_nt,
                            dim_ter=config.opts.dim_ter,
                            dim_act=config.opts.dim_act,
                            dropout=config.opts.dropout,
                            dim_h=config.opts.dim_h,
                            nn_layers=config.opts.nn_layers)

    if os.path.exists(config.opts.model):
        fname = 'epoch%03d.model' % (config.opts.epochs - 1)
        path = os.path.join(config.opts.model, fname)
        nnparser.load_mdl(path)

    print('testing...')
    idx = 0

    if os.path.exists(config.opts.model):
        fname = 'epoch%03d.model' % (config.opts.epochs - 1)
        path = os.path.join(config.opts.model, fname)
        nnparser.load_mdl(path)

    fname = 'test_res%s' % (tag)
    save_as = os.path.join(config.opts.result, fname)
    rf = open(save_as, 'w')

    contrast_vec=[]
    for word_tokens, act_tokens, term_tokens in data. \
            iter_data(data.word_tokens, data.act_tokens,
                      data.term_tokens, 'test'):
        output_actions, output_tokens = \
            nnparser.parse(word_tokens)

        pre_ans,pre_query=get_ans(output_actions,output_tokens)
        gold_ans,gold_query=get_ans(act_tokens,term_tokens)

        contrast=int(pre_ans==gold_ans)
        
        contrast_vec.append(contrast)
        print 'Current accuracy: %f' % ((np.sum(contrast_vec)+0.0)/len(contrast_vec))

        rf.write(pre_query +'\t'+gold_query+ '\n')

    rf.close()

    print 'Accuracy: %f' % ((np.sum(contrast_vec)+0.0)/len(contrast_vec))


