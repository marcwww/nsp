import operator

import dynet as dy

import ast
import data
import nnunits
import os
import config
import jpype

def recover_rlf(actions, terms):
# Recover the raw logic form from actions and terms
    output = ''
    for action in actions:
        if 'BEG' in action:
            output+='('
        elif 'RED' in action:
            output+=')'
        else:
            reverse='!' if 'SFTRR' in action else ''
            output+=reverse+terms.pop(0)+' '

    return output

class NNParser(object):

    def __init__(self,
                 word_dict=data.word_dict,
                 term_dict=data.term_dict,
                 dg_set=data.dg_set,
                 var_set=data.var_set,
                 date_set=data.date_set,
                 num_set=data.num_set,
                 entity_set=data.entity_set,
                 relation_set=data.relation_set,
                 act_dict=data.act_dict,
                 dim_word=config.opts.dim_word,
                 dim_term=config.opts.dim_term,
                 dim_act=config.opts.dim_act,
                 dim_h=config.opts.dim_h,
                 nn_layers=config.opts.nn_layers,
                 dropout=0.5,
                 pretrained_embedding=config.opts.emb,
                 attention_form='bilinear'):

        self._dropout=dropout

        self._pc=dy.ParameterCollection()

        self._word_dict = word_dict
        self._term_dict=term_dict
        self._term_set_map={'dg':dg_set,'var':var_set,
                            'var_lambda':set(var_set),
                            'date':date_set,'num':num_set,
                            'entity':entity_set,'relation':relation_set}
        self._act_dict = act_dict

        self._ACT_SFTG = self._filter_dg_acts(self._act_dict)
        self._ACT_SFTG_AND= self._act_dict['SFTG(and)']
        self._ACT_SFTN = self._act_dict['SFTN']
        self._ACT_SFTR = self._act_dict['SFTR']
        self._ACT_SFTRR = self._act_dict['SFTRR']
        self._ACT_BEG = self._act_dict['BEG']
        self._ACT_RED = self._act_dict['RED']


        self._word_in_layer= nnunits.LinearUnit(self._pc, dim_word, dim_h)
        self._term_in_layer= nnunits.LinearUnit(self._pc, dim_term, dim_h)
        self._term_dg_in_layer= nnunits.LinearUnit(self._pc, dim_term, dim_h)
        self._term_var_in_layer= nnunits.LinearUnit(self._pc, dim_term, dim_h)
        self._term_date_in_layer= nnunits.LinearUnit(self._pc, dim_term, dim_h)
        self._term_num_in_layer= nnunits.LinearUnit(self._pc, dim_term, dim_h)
        self._act_in_layer= nnunits.LinearUnit(self._pc, dim_act, dim_h)
        self._red_in_layer= nnunits.LinearUnit(self._pc, dim_h, dim_h)

        self._mlp_layer= nnunits.NonLinearUnit(self._pc, 4 * dim_h, dim_h)

        self._act_pred_layer= nnunits.LinearUnit(self._pc, dim_h, self._act_dict.size)
        self._term_pred_layer= nnunits.LinearUnit(self._pc, 2 * dim_h, self._term_dict.size)

        if attention_form=='bilinear':
            self._atten= nnunits.BiAttentionUnit(self._pc, dim_h, 2 * dim_h)
        elif attention_form=='feedforward':
            self._atten=nnunits.FFAttention(self._pc,dim_h,2*dim_h,dim_h)
        elif attention_form=='crf':
            self._atten=nnunits.FFAttention_Bernoulli(self._pc,dim_h,2*dim_h,dim_h)

        self._buffer_nn= nnunits.BiRNNSeqPredictor(dy.LSTMBuilder(nn_layers, dim_h,
                                                                  dim_h, self._pc))
        self._action_nn=dy.LSTMBuilder(1,dim_h,dim_h,self._pc)
        self._stack_nn=dy.LSTMBuilder(1,dim_h,dim_h,self._pc)
        self._init_embedding= nnunits.EmbeddingInitializer(self._pc, dim_h)

        self._lookup_word=self._pc.\
            add_lookup_parameters((self._word_dict.size,dim_word))
        self._lookup_term=self._pc.\
            add_lookup_parameters((self._term_dict.size,dim_term))
        self._lookup_act=self._pc.\
            add_lookup_parameters((self._act_dict.size,dim_act))

        self._load_pretrained_embedding(pretrained_embedding)

    def _load_pretrained_embedding(self,fpath):
        if fpath is not None:
            with open(fpath,'r') as f:
                for line in f:
                    line=line.strip().split(' ')
                    w,emb=line[0],[float(f) for f in line[1:]]
                    try:
                        w_idx=self._word_dict[w]
                        self._lookup_word.init_row(w_idx,emb)
                    except:
                        pass

    @property
    def root_idx(self):
        # This index refers the action 'GO' and term '<BOS>'
        return 0

    @property
    def SFTG_var(self):
        return self._act_dict['SFTG(var)']

    @property
    def STAGE_TRAIN(self):
        return 'stage_train'

    @property
    def STAGE_TEST(self):
        return 'stage_test'

    @property
    def NEST_DEPTH(self):
        return 10

    @property
    def pc(self):
        return self._pc

    def _filter_dg_acts(self,act_dict):
        dg_act=[]
        for str,idx in act_dict.str2idx.iteritems():
            if 'SFTG' in str:
                dg_act.append(idx)

        return dg_act

    def _enc_uttr(self,word_tokens):
        token_embs=[self._word_in_layer(self._lookup_word[self._word_dict[w_token]])
                    for w_token in word_tokens]

        forward_seq,back_ward_seq=self._buffer_nn.predict(token_embs)
        buffer=[dy.concatenate([f_states,b_states])
                for f_states,b_states in zip(forward_seq,reversed(back_ward_seq))]

        return buffer

    def train(self, word_tokens, superv_acts, superv_terms):
        dy.renew_cg()

        s_acts=list(superv_acts[1:])
        s_terms=list(superv_terms[1:])
        # Strip the 'GO' and '<BOS>'
        buffer=self._enc_uttr(word_tokens)
        act_tail=self._action_nn.initial_state()
        stack_tail=self._stack_nn.initial_state()
        act_tail=act_tail.add_input(self._act_in_layer(self._lookup_act[self.root_idx]))
        # Add 'GO'
        stack_tail=stack_tail.add_input(self._term_in_layer(self._lookup_term[self.root_idx]))
        # Add '<BOS>'

        loss=self._trans_loss(s_acts,s_terms,buffer,stack_tail,act_tail)

        return loss

    def _legal_acts(self,flags,cur_len,stage,nest_idx):

        acts=[]
        if stage==self.STAGE_TRAIN:
            # Training stage
            if cur_len>=0:
                acts += [self._ACT_BEG, self._ACT_SFTRR, self._ACT_SFTR, self._ACT_SFTN]\
                        + self._ACT_SFTG

            if cur_len>=2:
                acts += [self._ACT_RED]
        else:
            # Testing stage
            assert stage==self.STAGE_TEST, 'Stage is not stage_test'
            if cur_len>=0:
                acts += [self._ACT_BEG, self._ACT_SFTRR, self._ACT_SFTR, self._ACT_SFTN] \
                        + self._ACT_SFTG

                if nest_idx>self.NEST_DEPTH:
                    # Prevent into too deep
                    acts.remove(self._ACT_BEG)

                if flags['and']:
                    # For simplicity, do not allow nested 'and'
                    acts.remove(self._ACT_SFTG_AND)

                if flags['lambda'] and flags['var_lambda']:
                    # A lambda predicate has appeared and
                    #  the variable of it has not
                    acts = [self._ACT_SFTN]
                    return acts

                if not flags['lambda'] and self.SFTG_var in acts:
                    # If there has not been 'lambda',
                    # the 'var' can not exist either
                    acts.remove(self.SFTG_var)

                if not flags['BEG']:
                    # General predicates can only appear just after '('
                    for act in self._ACT_SFTG:
                        if act in acts:
                            acts.remove(act)
                    # Relation can only appear just after '('
                    acts.remove(self._ACT_SFTR)
                    acts.remove(self._ACT_SFTRR)
                else:
                    # The first term just after '(' cannot be an end node
                    acts.remove(self._ACT_SFTN)

                if flags['date'] or flags['num'] or flags['var']:
                    # There must be shift action after these general predicates
                    acts = [self._ACT_SFTN]
                    return acts
            if cur_len>=2:

                if not flags['lambda'] and flags['lambda_depth']>0:
                    # Now the lambda clause is over, it is needed to reduce the
                    #whole clause.
                    acts = [self._ACT_RED]
                    return acts

                acts += [self._ACT_RED]

                if flags['and'] and flags['and_nest_idx']==nest_idx:
                    if cur_len<3:
                        # print 'removed RED'
                        acts.remove(self._ACT_RED)
                    elif cur_len==3:
                        # print 'make RED'
                        acts=[self._ACT_RED]
                        return acts

        return acts

    def _trans_loss(self,superv_acts,superv_terms,
                    buffer,stack_tail,act_tail):
        stack=[]
        loss_lst=[]

        flags={'reduction':False,
               'var':False,
               'date':False,
               'num':False}
        nest_idx=1
        nest_len=0

        stack.append((stack_tail,self.root_idx,stack_tail))

        while not (len(stack)==2 and flags['reduction']!=False):
            flags['reduction']=False
            act_choices=self._legal_acts(flags,nest_len,self.STAGE_TRAIN,nest_idx)

            w_weights=None
            act=self._act_dict[superv_acts.pop(0)]

            # Accumlate loss in action predication
            if len(stack)>0 and act_choices[0]!=self._ACT_RED:
                stack_emb=stack[-1][0].output()
                act_emb=act_tail.output()
                w_weights=self._atten(stack_emb,buffer)
                # buf_emb,_=nnunits.attention_output(buffer,w_weights,'thre_sample')
                buf_emb, _ = nnunits.attention_output(buffer, w_weights, 'soft_average')
                # buf_emb=self._atten.output(buffer,w_weights)

                trans_state=dy.concatenate([buf_emb,stack_emb,act_emb])
                out=self._mlp_layer(trans_state)

                if self._dropout>0:
                    out=dy.dropout(out,self._dropout)

                if len(act_choices):
                    log_probs_act=dy.log_softmax(self._act_pred_layer(out),act_choices)
                    assert act in act_choices, 'illegal action'
                    loss_lst.append(-dy.pick(log_probs_act,act))

            act_emb=self._act_in_layer(self._lookup_act[act])
            act_tail=act_tail.add_input(act_emb)

            # Accumlate loss in term predication
            if act in [self._ACT_SFTN,self._ACT_SFTR,self._ACT_SFTRR]+self._ACT_SFTG:
                idx = self._term_dict[superv_terms.pop(0)]
                if w_weights is not None:
                    # buf_emb = self._atten.output(buffer, w_weights)
                    buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'soft_average')
                    log_probs_term=dy.log_softmax(self._term_pred_layer(buf_emb))
                    loss_lst.append(-dy.pick(log_probs_term,idx))

                stack_state,label,_=stack[-1] if stack else (stack_tail,self.root_idx,stack_tail)
                term_emb=self._term_in_layer(self._lookup_term[idx])
                # Here it is called 'raw embedding'

                stack_state=stack_state.add_input(term_emb)
                stack.append((stack_state,nest_idx,term_emb))
                # 'nl' label represents the non-leaf nodes

            elif act== self._ACT_BEG:
                nest_idx+=1

            else:
                assert act == self._ACT_RED, 'The action is not reduction'
                leaf_raw_reps=[]
                while stack[-1][1]==nest_idx:
                    top=stack.pop()
                    rep,_,raw_rep=top
                    leaf_raw_reps.append(raw_rep)

                nest_idx-=1

                subtree_rep=self._red_in_layer(dy.average(leaf_raw_reps))

                # Append the new reduced node
                stack_state, _, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                stack_state = stack_state.add_input(subtree_rep)
                stack.append((stack_state, nest_idx, subtree_rep))
                flags['reduction']=True

            nest_len=0
            for elem in reversed(stack):
                if elem[1] != nest_idx:
                    break
                nest_len += 1

        return dy.esum(loss_lst)

    def parse(self,word_tokens):

        dy.renew_cg()

        w_tokens = list(word_tokens)

        buffer = self._enc_uttr(word_tokens)
        act_tail = self._action_nn.initial_state()
        stack_tail = self._stack_nn.initial_state()
        act_tail = act_tail.add_input(self._act_in_layer(self._lookup_act[self.root_idx]))
        # Add 'GO'
        stack_tail = stack_tail.add_input(self._term_in_layer(self._lookup_term[self.root_idx]))
        # Add '<BOS>'

        return self._trans_run(w_tokens, buffer, stack_tail, act_tail)

    def _trans_run(self,word_tokens,buffer,stack_tail,act_tail):
        stack = []

        flags = {'reduction': False,
                 'BEG':False,
                 'var': False,
                 'lambda':False,
                 'var_lambda':False,
                 'date': False,
                 'num': False,
                 'and': False,
                 'and_depth':0,
                 'and_nest_idx':0,
                 'lambda_depth':0}

        out_actions=[]
        out_terms=[]

        inter_len=0
        nest_len = 0

        nest_idx=1
        stack.append((stack_tail, self.root_idx, stack_tail))

        while flags['reduction'] == False or len(stack) != 2:

            flags['reduction'] = False
            act_choices = self._legal_acts(flags,nest_len,self.STAGE_TEST,nest_idx)

            # The flag is invalid again
            flags['BEG'] = False

            w_weights = None
            # act = act_choices[0]
            act=None

            # Predict action
            if len(act_choices)>=1:
                stack_emb = stack[-1][0].output()
                act_emb = act_tail.output()
                w_weights = self._atten(stack_emb, buffer)
                # buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'thre_sample')
                buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'soft_average')
                # buf_emb = self._atten.output(buffer, w_weights)

                trans_state = dy.concatenate([buf_emb, stack_emb, act_emb])
                out = self._mlp_layer(trans_state)
                log_probs_act = dy.log_softmax(self._act_pred_layer(out), act_choices)
                act = max(enumerate(log_probs_act.vec_value()),
                         key=operator.itemgetter(1))[0]
                assert act in act_choices, 'illegal action'

            act_emb = self._act_in_layer(self._lookup_act[act])
            act_tail = act_tail.add_input(act_emb)

            # Predicate term
            if act in [self._ACT_SFTN,self._ACT_SFTR, self._ACT_SFTRR]:
                filter_set=None
                filter_key=None
                for key,flag in flags.items():
                    if flag and key in self._term_set_map:
                        filter_set=self._term_set_map[key]
                        filter_key=key
                        break

                if w_weights is not None:
                    buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'soft_average')
                    # _, wids = nnunits.attention_output(buffer, w_weights, 'hard_sample')
                    # buf_emb = self._atten.output(buffer, w_weights)
                    # buf_emb,wid = nnunits.hard_sample(buffer,w_weights)
                    # _,wids=nnunits.threshold_sample(buffer,w_weights)

                    # if wids:
                    #     aligned_words=[]
                    #     for id in [wids]:
                    #         aligned_words.append(word_tokens[id])
                    #     print aligned_words

                    log_probs_term = dy.log_softmax(self._term_pred_layer(buf_emb))

                    if filter_set:
                        assert act not in [self._ACT_SFTR, self._ACT_SFTRR], \
                            'The action should not be shifting a relation'
                        # Target-specific shift
                        idx=max([(i,log_probs_term.vec_value()[i]) for i in filter_set],
                             key=operator.itemgetter(1))[0]
                        if filter_key=='var':
                            assert flags['lambda']==True, 'There has no lambda before'
                            # The lambda clause can be closed now
                            flags['lambda']=False
                    else:
                        if act == self._ACT_SFTN:
                            filter_set=self._term_set_map['entity']
                        else:
                            filter_set=self._term_set_map['relation']

                        idx = max([(i, log_probs_term.vec_value()[i]) for i in filter_set],
                                  key=operator.itemgetter(1))[0]

                if filter_key:
                    flags[filter_key]=False

                stack_state, label, _ = stack[-1] if stack \
                    else (stack_tail, self.root_idx, stack_tail)
                term_emb = self._term_in_layer(self._lookup_term[idx])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(term_emb)
                stack.append((stack_state, nest_idx, term_emb))
                # 'nl' label represents the non-leaf nodes

                out_actions.append(self._act_dict.idx2str[act])
                out_terms.append(self._term_dict.idx2str[idx])

                inter_len+=1

            elif act in self._ACT_SFTG:
                if 'date' in self._act_dict.idx2str[act]:
                    flags['date']=True
                elif 'num' in self._act_dict.idx2str[act]:
                    flags['num']=True
                elif 'var' in self._act_dict.idx2str[act]:
                    flags['var']=True
                elif 'lambda' in self._act_dict.idx2str[act]:
                    flags['var_lambda']=True
                    flags['lambda']=True
                elif 'and' in self._act_dict.idx2str[act]:
                    flags['and']=True
                    flags['and_nest_idx']=int(nest_idx)

                term_dg=self._act_dict.idx2str[act].rstrip(')').lstrip('SFTG(')
                idx = self._term_dict[term_dg]
                # There is no terms (operands) for this action

                stack_state, label, _ = stack[-1] if stack else (stack_tail, self.root_idx, stack_tail)
                term_dg_emb = self._term_dg_in_layer(self._lookup_term[idx])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(term_dg_emb)
                stack.append((stack_state, nest_idx, term_dg_emb))
                # 'nl' label represents the non-leaf nodes

                out_actions.append(self._act_dict.idx2str[act])
                out_terms.append(self._term_dict.idx2str[idx])

                inter_len += 1

            elif act== self._ACT_BEG:
                nest_idx+=1
                out_actions.append(self._act_dict.idx2str[act])
                flags['BEG'] = True
                if flags['lambda']:
                    # In the lambda clause, record the nest depth
                    flags['lambda_depth']+=1
                # Showing it is into a new nested structure,
                #and the flag is invalid when a new shift happens

            else:
                if flags['and'] and flags['and_nest_idx'] == nest_idx:
                    assert nest_len == 3, 'The nest length is not 3 for dg pre and'
                    flags['and']=False

                assert act==self._ACT_RED, 'The action is not reduction'
                leaf_raw_reps = []
                while stack[-1][1] == nest_idx:
                    top = stack.pop()
                    rep, _, raw_rep = top
                    leaf_raw_reps.append(raw_rep)
                nest_idx-=1

                subtree_rep = self._red_in_layer(dy.average(leaf_raw_reps))

                # Append the new reduced node
                stack_state, _, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                stack_state = stack_state.add_input(subtree_rep)
                stack.append((stack_state, nest_idx, subtree_rep))
                flags['reduction'] = True
                if not flags['lambda'] and flags['lambda_depth']>0:
                    # Reduce action decrement the nest depth for the
                    #lambda clause
                    flags['lambda_depth']-=1

                out_actions.append(self._act_dict.idx2str[act])

            # Caculate the length of current nested structure
            nest_len=0
            for elem in reversed(stack):
                if elem[1]!=nest_idx:
                    break
                nest_len+=1

            if len(out_terms)+len(out_actions)>len(word_tokens)*20:
                break

        return out_actions,out_terms

    def save_mdl(self,fpath):
        self._pc.save(fpath)

    def load_mdl(self,fpath):
        self._pc.populate(fpath)

if __name__ == '__main__':
    data.load()

    lr=ast.MRL('(count (!fb:tv.tv_producer_term.program ((lambda x (fb:tv.tv_producer_term.producer (var x))) fb:en.danny_devito)))')
    lr.parse()
    terms,acts,_,_,_=lr.terms_and_acts(data.dg_predicates)

    print recover_rlf(acts,terms)

    # lr = ast.LogicRep('answer(count(intersection(city(cityid(\'austin\', _)), loc_2(countryid(\'usa\')))))')
    # # lr=LogicRep('answer(state(loc_2(countryid(CountryName))))')
    # # print(lr.tokens)
    # # print(lr.reconvert())
    # nlst = lr.parse()
    # # print nlst
    # ast = ast.Tree(nlst)
    # # print ast.get_nlves_and_lves()
    # gp_predicates = data.gp_predicates
    # terms,acts=ast.get_terms_and_acts(gp_predicates)
    # print terms
    # print acts
    # tm=trans_machine(terms,acts)
    # print tm.run()