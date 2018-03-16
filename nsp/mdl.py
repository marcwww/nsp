import operator

import dynet as dy

import ast
import data
import nnunits
import config


class trans_machine(object):

    def __init__(self,terms,acts):
        self._stack=[]
        self._buffer=[]
        self._terms=terms
        self._acts=acts

    def _action_type(self,action):
        if action.find('TER')==0:
            return 'TER'
        elif action.find('RED')==0:
            return 'RED'
        elif action.find('NT(')==0:
            return 'NT_G'
        else:
            return 'NT'

    def _reduction(self):
        terms=[]
        while self._stack[-1][0]=='l':
            terms.append(self._stack.pop())

        nl_node=self._stack.pop()
        self._stack.append(('l',nl_node[1]))
        return terms


    def run(self):
        self._buffer=list(self._terms)
        actions=list(self._acts)

        reduction_flag=False
        while not (len(self._stack)==1 and reduction_flag):
            reduction_flag=False
            action=actions.pop(0)
            atype=self._action_type(action)
            if atype=='TER':
                self._stack.append(('l',self._buffer.pop(0)))
            elif atype=='NT':
                self._stack.append(('nl',self._buffer.pop(0)))
            elif atype=='NT_G':

                self._stack.append(('nl',self._buffer.pop(0)))
            else:
                self._reduction()
                reduction_flag=True

        if len(self._stack)==1 and len(self._buffer)==0 and reduction_flag:
            return True
        else:
            return False

def recover_rlf(actions, terms):
# Recover the raw logic form from  actions and terms
    output = ''
    for action in actions:
        if 'NT' in action:
            nt = terms.pop(0)
            output += '( ' + nt+' '
        elif action == 'TER':
            ter = terms.pop(0)
            output += ter+' '
        else:
            output += ') '
    return output

class NNParser(object):

    def __init__(self, word_dict,nt_dict,ter_dict,act_dict,
                 dim_word,dim_nt,dim_ter,dim_act,dim_h,nn_layers,
                 dropout=0.5,
                 # pretrained_embedding=config.opts.emb,
                 pretrained_embedding=None,
                 attention_form='bilinear'):
        self._dropout=0.5

        self._pc=dy.ParameterCollection()

        self._word_dict=word_dict
        self._nt_dict=nt_dict
        self._ter_dict=ter_dict
        self._act_dict=act_dict

        self._ACT_NT_dg=self._filter_dg_acts(self._act_dict)
        self._ACT_NT=self._act_dict['NT']
        self._ACT_TER=self._act_dict['TER']
        self._ACT_RED=self._act_dict['RED']

        self._word_in_layer= nnunits.LinearUnit(self._pc, dim_word, dim_h)
        self._nt_in_layer= nnunits.LinearUnit(self._pc, dim_nt, dim_h)
        self._ter_in_layer= nnunits.LinearUnit(self._pc, dim_ter, dim_h)
        self._act_in_layer= nnunits.LinearUnit(self._pc, dim_act, dim_h)
        self._red_in_layer= nnunits.LinearUnit(self._pc, 2 * dim_h, dim_h)

        self._mlp_layer= nnunits.NonLinearUnit(self._pc, 5 * dim_h, dim_h)

        self._act_pred_layer= nnunits.LinearUnit(self._pc, dim_h, act_dict.size)
        self._nt_pred_layer= nnunits.LinearUnit(self._pc, 2 * dim_h, nt_dict.size)
        self._ter_pred_layer= nnunits.LinearUnit(self._pc, 2 * dim_h, ter_dict.size)

        if attention_form=='bilinear':
            self._atten= nnunits.BiAttentionUnit(self._pc, dim_h, 2 * dim_h)
        elif attention_form == 'feedforward':
            self._atten = nnunits.FFAttention(self._pc, dim_h, 2 * dim_h, dim_h)

        self._buffer_nn= nnunits.BiRNNSeqPredictor(dy.LSTMBuilder(nn_layers, dim_h,
                                                                  dim_h, self._pc))
        self._action_nn=dy.LSTMBuilder(1,dim_h,dim_h,self._pc)
        self._stack_nn=dy.LSTMBuilder(1,dim_h,dim_h,self._pc)
        self._init_embedding= nnunits.EmbeddingInitializer(self._pc, dim_h)

        self._lookup_word=self._pc.\
            add_lookup_parameters((self._word_dict.size,dim_word))
        self._lookup_nt=self._pc.\
            add_lookup_parameters((self._nt_dict.size,dim_nt))
        self._lookup_ter=self._pc.\
            add_lookup_parameters((self._ter_dict.size,dim_ter))
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
        return 0

    @property
    def pc(self):
        return self._pc

    def _filter_dg_acts(self,act_dict):
        dg_act=[]
        for str,idx in act_dict.str2idx.iteritems():
            if 'NT' in str and str!='NT':
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

        s_acts=list(superv_acts)
        s_terms=list(superv_terms)
        buffer=self._enc_uttr(word_tokens)
        act_tail=self._action_nn.initial_state()
        stack_tail=self._stack_nn.initial_state()

        loss=self._trans_loss(s_acts,s_terms,buffer,stack_tail,act_tail)

        return loss

    def _legal_acts(self,stack,reducable_flag,nt_flag=None,ter_flag=None):
        acts=[]
        if len(stack)==0:
            acts+=[self.root_idx]
        if len(stack)>=1:
            if nt_flag==None and ter_flag==None:
                acts+=[self._ACT_TER,self._ACT_NT]+self._ACT_NT_dg
            else:
                if ter_flag==True:
                    acts+=[self._ACT_TER]
                if nt_flag==True:
                    acts+=[self._ACT_NT]+self._ACT_NT_dg
        if len(stack)>=2 and reducable_flag !=False:
            acts+=[self._ACT_RED]

        return acts

    def _trans_loss(self,superv_acts,superv_terms,
                    buffer,stack_tail,act_tail):
        stack=[]
        loss_lst=[]

        reduction_flag=False
        reducable_flag=False
        while not (len(stack)==1 and reduction_flag!=False):
            reduction_flag=False
            act_choices=self._legal_acts(stack,reducable_flag)

            w_weights=None
            act=self._act_dict[superv_acts.pop(0)]

            # Accumlate loss in action predication
            if len(stack)>0 and act_choices[0]!=self._ACT_RED:
                stack_emb=stack[-1][0].output()
                act_emb=act_tail.output()
                w_weights=self._atten(stack_emb,buffer)
                buf_emb,_=nnunits.attention_output(buffer,w_weights,'soft_average')
                # buf_emb=self._atten.output(buffer,w_weights)

                for i in xrange(len(stack)):
                    re_idx=len(stack)-1-i
                    if stack[re_idx][1]=='nl':
                        nl_emb=stack[re_idx][2]
                        # Find the raw embedding of the root of subtree
                        # for the leaves.
                        break

                trans_state=dy.concatenate([buf_emb,stack_emb,nl_emb,act_emb])
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
            if act==self._ACT_NT:
                idx_nt=self._nt_dict[superv_terms.pop(0)]
                if w_weights is not None:
                    buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'soft_average')
                    # buf_emb = self._atten.output(buffer, w_weights)
                    log_probs_nt=dy.log_softmax(self._nt_pred_layer(buf_emb))
                    loss_lst.append(-dy.pick(log_probs_nt,idx_nt))

                stack_state,label,_=stack[-1] if stack else (stack_tail,'ROOT',stack_tail)
                nt_emb=self._nt_in_layer(self._lookup_nt[idx_nt])
                # Here it is called 'raw embedding'

                stack_state=stack_state.add_input(nt_emb)
                stack.append((stack_state,'nl',nt_emb))
                # 'nl' label represents the non-leaf nodes

            elif act in self._ACT_NT_dg:
                idx_nt=self._nt_dict[superv_terms.pop(0)]
                # There is no terms (operands) for this action

                stack_state, label, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                nt_emb = self._nt_in_layer(self._lookup_nt[idx_nt])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(nt_emb)
                stack.append((stack_state, 'nl', nt_emb))
                # 'nl' label represents the non-leaf nodes

            elif act==self._ACT_TER:
                idx_ter=self._ter_dict[superv_terms.pop(0)]
                if buf_emb!=None:
                    log_probs_ter=dy.log_softmax(self._ter_pred_layer(buf_emb))
                    loss_lst.append(-dy.pick(log_probs_ter,idx_ter))

                stack_state, label, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                ter_emb = self._nt_in_layer(self._lookup_ter[idx_ter])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(ter_emb)
                stack.append((stack_state, 'l', ter_emb))
                # 'nl' label represents the non-leaf nodes

            else:
                leaf_raw_reps=[]
                while stack[-1][1]=='l':
                    top=stack.pop()
                    rep,_,raw_rep=top
                    leaf_raw_reps.append(raw_rep)

                nl_raw_rep=stack.pop()[2]
                subtree_rep=self._red_in_layer(
                    dy.concatenate([dy.average(leaf_raw_reps),nl_raw_rep]))

                # Append the new reduced node
                stack_state, _, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                stack_state = stack_state.add_input(subtree_rep)
                stack.append((stack_state, 'l', subtree_rep))
                reduction_flag=True

            reducable_flag=True if stack[-1][1]!='nl' else False

        # for loss in loss_lst:
        #     print loss.vec_value()


        return dy.esum(loss_lst)

    def _trans_run(self,word_tokens,buffer,stack_tail,act_tail):
        stack = []

        reduction_flag = False
        reducable_flag = False

        nt_flag=True
        ter_flag=True

        out_actions=[]
        out_terms=[]

        nt_num=0

        while reduction_flag == False or len(stack) != 1:
            reduction_flag = False
            act_choices = self._legal_acts(stack,reducable_flag,nt_flag,ter_flag)

            w_weights = None
            act = act_choices[0]

            # Predict action
            if (len(stack) > 0 and act_choices[0] != self._ACT_RED) \
                    or len(act_choices)>1:
                stack_emb = stack[-1][0].output()
                act_emb = act_tail.output()
                w_weights = self._atten(stack_emb, buffer)
                # buf_emb = self._atten.output(buffer, w_weights)
                buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'soft_average')

                for i in xrange(len(stack)):
                    re_idx = len(stack) - 1 - i
                    if stack[re_idx][1] == 'nl':
                        nl_emb = stack[re_idx][2]
                        # Find the raw embedding of the root of subtree
                        # for the leaves.
                        break

                trans_state = dy.concatenate([buf_emb, stack_emb, nl_emb, act_emb])
                out = self._mlp_layer(trans_state)
                log_probs_act = dy.log_softmax(self._act_pred_layer(out), act_choices)
                act =max(enumerate(log_probs_act.vec_value()),
                         key=operator.itemgetter(1))[0]
                assert act in act_choices, 'illegal action'

            act_emb = self._act_in_layer(self._lookup_act[act])
            act_tail = act_tail.add_input(act_emb)

            # Predicate term
            if act == self._ACT_NT:
                if w_weights is not None:
                    # buf_emb = self._atten.output(buffer, w_weights)
                    buf_emb,_ = nnunits.attention_output(buffer, w_weights, 'soft_average')

                    log_probs_nt = dy.log_softmax(self._nt_pred_layer(buf_emb))
                    idx_nt=max(enumerate(log_probs_nt.vec_value()),
                         key=operator.itemgetter(1))[0]

                stack_state, label, _ = stack[-1] if stack \
                    else (stack_tail, 'ROOT', stack_tail)
                nt_emb = self._nt_in_layer(self._lookup_nt[idx_nt])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(nt_emb)
                stack.append((stack_state, 'nl', nt_emb))
                # 'nl' label represents the non-leaf nodes

                out_actions.append(self._act_dict.idx2str[act])
                out_terms.append(self._nt_dict.idx2str[idx_nt])
                nt_num+=1

            elif act in self._ACT_NT_dg:
                nt=self._act_dict.idx2str[act].rstrip(')').lstrip('NT(')
                idx_nt = self._nt_dict[nt]
                # There is no terms (operands) for this action

                stack_state, label, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                nt_emb = self._nt_in_layer(self._lookup_nt[idx_nt])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(nt_emb)
                stack.append((stack_state, 'nl', nt_emb))
                # 'nl' label represents the non-leaf nodes

                out_actions.append(self._act_dict.idx2str[act])
                out_terms.append(self._nt_dict.idx2str[idx_nt])
                nt_num+=1

            elif act == self._ACT_TER:
                if buf_emb!=None:
                    log_probs_ter = dy.log_softmax(self._ter_pred_layer(buf_emb))
                    idx_ter = max(enumerate(log_probs_ter.vec_value()),
                                  key=operator.itemgetter(1))[0]

                stack_state, label, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                ter_emb = self._nt_in_layer(self._lookup_ter[idx_ter])
                # Here it is called 'raw embedding'

                stack_state = stack_state.add_input(ter_emb)
                stack.append((stack_state, 'l', ter_emb))
                # 'nl' label represents the non-leaf nodes

                out_actions.append(self._act_dict.idx2str[act])
                out_terms.append(self._ter_dict.idx2str[idx_ter])

            else:
                leaf_raw_reps = []
                while stack[-1][1] == 'l':
                    top = stack.pop()
                    rep, _, raw_rep = top
                    leaf_raw_reps.append(raw_rep)

                nl_raw_rep = stack.pop()[2]
                subtree_rep = self._red_in_layer(
                    dy.concatenate([dy.average(leaf_raw_reps), nl_raw_rep]))

                # Append the new reduced node
                stack_state, _, _ = stack[-1] if stack else (stack_tail, 'ROOT', stack_tail)
                stack_state = stack_state.add_input(subtree_rep)
                stack.append((stack_state, 'l', subtree_rep))
                reduction_flag = True

                out_actions.append(self._act_dict.idx2str[act])

            reducable_flag = True if stack[-1][1] != 'nl' else False
            nt_flag=True
            ter_flag=True

            nl_num=0
            for elem in stack:
                if elem[1]=='nl':
                    nl_num+=1
            if nl_num>=10 or len(stack)>len(word_tokens)\
                    or nt_num>len(word_tokens):
                nt_flag=False

            l_num=0
            for elem in stack[::-1]:
                if elem[1]=='l':
                    l_num+=1
                else:
                    break
            if l_num>=10:
                ter_flag=False

        return out_actions,out_terms

    def parse(self,word_tokens):

        dy.renew_cg()

        w_tokens = list(word_tokens)

        buffer = self._enc_uttr(word_tokens)
        act_tail = self._action_nn.initial_state()
        stack_tail = self._stack_nn.initial_state()

        return self._trans_run(w_tokens, buffer, stack_tail, act_tail)


    def save_mdl(self,fpath):
        self._pc.save(fpath)

    def load_mdl(self,fpath):
        self._pc.populate(fpath)



if __name__ == '__main__':
    data.load()

    lr = ast.LogicRep('answer(count(intersection(city(cityid(\'austin\', _)), loc_2(countryid(\'usa\')))))')
    # lr=LogicRep('answer(state(loc_2(countryid(CountryName))))')
    # print(lr.tokens)
    # print(lr.reconvert())
    nlst = lr.parse()
    # print nlst
    ast = ast.Tree(nlst)
    # print ast.get_nlves_and_lves()
    gp_predicates = data.gp_predicates
    terms,acts=ast.get_terms_and_acts(gp_predicates)
    print terms
    print acts
    tm=trans_machine(terms,acts)
    print tm.run()