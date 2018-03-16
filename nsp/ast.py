import re

import data

class MRL(object):
    def __init__(self,raw_lf):
        raise NotImplementedError('MRL.__init__: not implemented')

    @property
    def tokens(self):
        raise NotImplementedError('MRL.tokens: not implemented')


    def parse(self):
        raise NotImplementedError('MRL.parse: not implemented')


class LogicRep(MRL):
    def __init__(self,raw_lf):
        self._raw_lf=raw_lf
        self._tokens=self._tokenize()
        if self._tokens[0]!='(':
            self._convert()

    def _tokenize(self):
        tokens = re.split('(\s+|\(|\))', str(self._raw_lf))

        for tok_idx in reversed(range(len(tokens))):
            w=tokens[tok_idx]
            if w.isspace():
                tokens.remove(w)

        for tok_idx in reversed(range(len(tokens))):
            w=tokens[tok_idx]

            if len(w)==0:
                continue
            if w[-1]=='\'' and w[0]!='\'':
                w_pre=tokens[tok_idx-1]
                if w_pre[-1]!='\'' and w_pre[0]=='\'':
                    tokens[tok_idx]=' '.join([w_pre,w])
                    tokens.remove(w_pre)

        # tokens = re.split('(\,|\(|\))', str(self._raw_lf))
        # print [t for t in tokens if len(t) and not t.isspace()]
        return [t for t in tokens if len(t) and not t.isspace()]

    @property
    def tokens(self):
        return self._tokens

    def _convert(self):
    # Convert logic form, from 'a(b(c),d(e))' to '(a (b c),(d e))'
        idx=0
        while idx<len(self._tokens)-1:
            if self._tokens[idx]!='(' and self._tokens[idx+1]=='(':
                self._tokens[idx],self._tokens[idx+1]=\
                    self._tokens[idx+1],self._tokens[idx]
                idx+=2
            else:
                idx+=1

    def reconvert(self):
    # Inverse procedure of _convert()
        tokens=list(self._tokens)
        tid = 0
        while tid < len(tokens) - 1:
            if tokens[tid] == '(' and tokens[tid + 1] != '(':
                tokens[tid], tokens[tid + 1] = tokens[tid + 1], tokens[tid]
                tid += 2
            else:
                tid += 1

        return tokens

    def _parse(self,tokens):

        assert len(tokens) and tokens[0]=='('
        ast_nlst=[]
        # Ast node list

        tokens.pop(0)
        # Pop the head '('

        operator=tokens.pop(0)
        operands, tokens=self._parse_operands(tokens)
        ast_nlst.append(operator)
        ast_nlst.extend(operands)

        assert len(tokens) and tokens[0]==')'
        tokens.pop(0)
        # Pop the tail ')'

        return ast_nlst,tokens

    def _parse_operands(self,tokens):
        operands=[]

        while tokens[0]!=')':
            # Operand type one: nested operand
            if tokens[0]=='(':
                ast_nlst,tokens=self._parse(tokens)
                operands.append(ast_nlst)
            else:
                operands.append(tokens.pop(0))

        return operands,tokens

    def parse(self):
    # Parse logic representation from long-string format into
    #that in nested list format
        ast_nlst,_=self._parse(self._tokens)

        return ast_nlst

    def standard_output(self):
        lf = self.reconvert()
        ' '.join(lf)
        return lf.rstrip().\
            replace('( ', '(').replace(' )', ')')



class Node(object):

    def __init__(self,idx,str):
        self._idx=idx
        self._str=str
        self._children=[]

    @property
    def idx(self):
        return self._idx

    @property
    def str(self):
        return self._str

    @property
    def children(self):
        return self._children

    def append_child(self,idx):
        self._children.append(idx)

class Tree(object):
    def __init__(self,nlst):
        self._node_dict={}
        self._node_num=0
        self._build_from_nlst(nlst)

    @property
    def root_idx(self):
        return 0

    @property
    def node_dict(self):
        return self._node_dict

    def _gen_node(self,str):
        idx=self._node_num
        node=Node(idx,str)
        self._node_dict[idx]=node
        self._node_num+=1
        return idx

    # def append_node(self,str,pidx=None):
    #     idx=self._gen_node(str)
    #
    #     if pidx is not None:
    #         self._node_dict[pidx].append_child(idx)

    def _build_from_nlst(self,nlst):
    # Build ast from nested list
        def build_from(obj):
            if isinstance(obj,basestring):
                return self._gen_node(obj)
            else:
                assert isinstance(obj,list)
                idx=self._gen_node(obj[0])
                for c_obj in obj[1:]:
                    c_idx=build_from(c_obj)
                    self._node_dict[idx].append_child(c_idx)
                return idx

        build_from(nlst)

    def _build_from_lr(self,lr):
        nlst=lr.parse()
        self._build_from_nlst(nlst)

    def get_nlves_and_lves(self):
    # Get non-leaf nodes(non terminals) and leaf nodes(terminals)
        nlves=[]
        lves=[]
        for idx,node in self._node_dict.iteritems():
            if node.children:
                nlves.append(node.str)
            else:
                lves.append(node.str)
        return nlves,lves

    def get_terms_and_acts(self,dg_predicates):
    # DFS in the AST, get the actions and terms which
    #indicate the generation process of the AST
    # dg_predicates: domain-general predicates
    # RED: reduction
    # NT: into a new branch
    # TER: leaf node
        acts=[]
        terms=[]

        def top_down_analyze(idx):
            node=self._node_dict[idx]
            terms.append(node.str)

            if node.children:
                action='NT'
                if node.str in dg_predicates:
                    action+='({})'.format(node.str)
                acts.append(action)

                for c_idx in node.children:
                    top_down_analyze(c_idx)
                acts.append('RED')
            else:
                acts.append('TER')

        top_down_analyze(self.root_idx)

        return terms,acts


if __name__ == '__main__':
    data.load()

    # lr=LogicRep('answer(count(intersection(city(cityid(\'austin\', _)),'
    #             ' loc_2(countryid(\'usa\')))))')
    lr=LogicRep('answer(state(loc_2(countryid(CountryName))))')
    # lr=LogicRep('(count (!fb:tv.tv_producer_term.program ((lambda x (fb:tv.tv_producer_term.producer (var x))) fb:en.danny_devito)))')
    print(lr.tokens)
    print(lr.reconvert())
    nlst=lr.parse()
    print nlst
    ast=Tree(nlst)
    print ast.get_nlves_and_lves()
    gp_predicates= data.gp_predicates
    print ast.get_terms_and_acts(gp_predicates)[0]
    print ast.get_terms_and_acts(gp_predicates)[1]



