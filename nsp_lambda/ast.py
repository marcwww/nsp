import re
import os
import data
import config
import jpype

class MRL(object):
    def __init__(self,raw_lf):
        self._raw_lf = raw_lf
        self._tokens = self._tokenize()

        self._LispTree = jpype.JClass('fig.basic.LispTree')
        self._lispTree = None

    def _tokenize(self):
        tokens = re.split('(\s+|\(|\))', str(self._raw_lf))
        return [t for t in tokens if len(t) and not t.isspace()]

    @property
    def tokens(self):
        return self._tokens

    def parse(self):
        self._lispTree=self._LispTree.proto.parseFromString(self._raw_lf)

        return self._lispTree

    def _is_num(self,str):
        try:
            int(str)
        except:
            try:
                float(str)
            except:
                return False

        return True

    def terms_and_acts(self,dg_predicates):
        acts=[]
        terms=[]
        variables=[]
        nums=[]
        dates=[]
        dgs=[]
        entitys=[]
        relations=[]
        flags={'date':False,
               'number':False,
               'var':False,
               'end_node':False,
               'lambda':False}

        def top_down(root,flags):

            if root.children:
                acts.append('BEG')

                for idx in range(len(root.children)):
                    flags['end_node']=(idx==len(root.children)-1 or\
                                       root.children[idx-1].toString()=='lambda')
                    for child in root.children:
                        if child.toString() in ['date','number','var'] and\
                            child.toString() != root.children[idx].toString():
                            flags['end_node']=True
                            break

                    top_down(root.children[idx],flags)

                # Inspect the subtree for types 'data',
                # 'number' and 'var':
                if flags['date']==True:
                    dd_mm_yyyy=[terms.pop() for _ in range(3)]
                    [acts.pop() for _ in range(3-1)]
                    date=' '.join(reversed(dd_mm_yyyy))
                    terms.append(date)
                    dates.append(date)
                    flags['date']=False

                elif flags['number']==True:
                    num_unit=[terms.pop() for _ in range(2)]
                    [acts.pop() for _ in range(2-1)]
                    num=' '.join(reversed(num_unit))
                    terms.append(num)
                    nums.append(num)
                    flags['number']=False

                elif flags['var']:
                    variables.append(terms[-1])
                    flags['var']=False

                acts.append('RED')
            else:
                term=root.toString()
                if term in dg_predicates:
                    act='SFTG(%s)' % (term)
                    acts.append(act) # Shift in a domain-general predicate
                    terms.append(term)
                    dgs.append(term)

                    # For term date, number and var
                    if term in flags.keys():
                        flags[term]=True
                else:
                    if flags['end_node']:
                        acts.append('SFTN')
                        terms.append(term)  # Shift in an end node(not including number,date and x)
                        if not flags['date'] and not flags['number'] and\
                                not flags['var'] and not flags['lambda']:
                            entitys.append(term)
                        if flags['lambda']:
                            # Unset the flag 'lambda' if the x has just been recorded
                            flags['lambda']=False

                    elif term[0]=='!':
                        acts.append('SFTRR')
                        terms.append(term[1:])  # Shift in the reverse of a relation
                        if not flags['date'] and not flags['number'] and not flags['var']:
                            relations.append(term[1:])
                    else:
                        acts.append('SFTR')
                        terms.append(term)  # Shift in a relation
                        if not flags['date'] and not flags['number'] and not flags['var']:
                            relations.append(term)

        top_down(self._lispTree,flags)
        return terms,acts,dates,nums,variables,dgs,entitys,relations

    def term_and_act_pairs(self,terms,acts):
        pairs=[]
        delta = 0
        for i in range(len(acts)):
            if acts[i] != 'RED' and acts[i] != 'BEG':
                pairs.append((acts[i], terms[i - delta]))
            else:
                pairs.append((acts[i],''))
                delta += 1

        return pairs

if __name__ == '__main__':
    jar_path = os.path.join(config.opts.lib, 'lDCS_convertor.jar')
    jpype.startJVM(jpype.getDefaultJVMPath(), '-ea',
                   '-Djava.class.path=%s' % jar_path)
    data.load()

    # lr=MRL('(count (!fb:tv.tv_producer_term.program ((lambda x (fb:tv.tv_producer_term.producer (var x))) fb:en.danny_devito)))')
    # lr=MRL('(!fb:award.award_honor.award (and ((lambda x (fb:award.award_honor.year (var x))) (date 1968 -1 -1)) '
    #        '((lambda x (fb:award.award_honor.award_winner (var x))) fb:en.jack_albertson)))')
    # lr=MRL('(fb:government.us_president.presidency_number (number 22.0 fb:en.unitless))')
    lr=MRL('(!fb:architecture.ownership.end_date (and ((lambda x (fb:architecture.ownership.owner (var x))) fb:en.john_j_raskob) ((lambda x (fb:architecture.ownership.structure (var x))) fb:en.empire_state_building)))')
    print(lr.tokens)
    lr.parse()
    terms, acts, dates, nums, variables, dgs, entitys, relations\
        = lr.terms_and_acts(data.dg_predicates)
    print 'TERMS:',terms
    print 'ENTITIES:',entitys
    print 'RELATIONS:',relations

    for line in lr.term_and_act_pairs(terms,acts):
        print line

    jpype.shutdownJVM()

