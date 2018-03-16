import dynet as dy
import numpy as np
from operator import itemgetter

class EmbeddingInitializer:
    def __init__(self,pc,dim):
        self._init_embedding=pc.add_parameters((dim,))

    def __call__(self):
        return dy.parameter(self._init_embedding)

class LinearUnit:
    def __init__(self,pc,dim_in,dim_out):
        self.W=pc.add_parameters((dim_out,dim_in))
        self.b=pc.add_parameters((dim_out,))

    def __call__(self,x):
        W=dy.parameter(self.W)
        b=dy.parameter(self.b)
        return W*x+b

class NonLinearUnit:
    def __init__(self,pc,dim_in,dim_out,act_func=dy.tanh):
        self.act_func=act_func
        self.W=pc.add_parameters((dim_out,dim_in))
        self.b=pc.add_parameters((dim_out))

    def __call__(self,x):
        W=dy.parameter(self.W)
        b=dy.parameter(self.b)
        return self.act_func(W*x+b)

class SeqPredictor:
    def __init__(self):
        pass
    def predict(self,inputs):
        raise NotImplementedError('SeqPredictor.predict: not implemented')

class BiRNNSeqPredictor(SeqPredictor):
    def __init__(self,lstm_builder):
        self.forward_builder=lstm_builder
        self.backward_builder=lstm_builder
    def predict(self,inputs):
        forward_init=self.forward_builder.initial_state()
        backward_init=self.backward_builder.initial_state()
        forward_seq=[x.output() for x in forward_init.add_inputs(inputs)]
        backward_seq=[x.output() for x in forward_init.add_inputs(reversed(inputs))]

        return forward_seq,backward_seq

class BiAttentionUnit:
    def __init__(self,pc,dim_dec,dim_enc):
        self.W=pc.add_parameters((dim_dec,dim_enc))

    def __call__(self,state_dec,states_enc):
        w=dy.parameter(self.W)
        a_weights=[]
        for state_enc in states_enc:
            a_w=dy.dot_product(w*state_enc,state_dec)
            a_weights.append(a_w)

        a_weights=dy.softmax(dy.concatenate(a_weights))
        return a_weights

    def output(self,states_enc,a_weights):
        return dy.esum([vec*a_weight for vec, a_weight in zip(states_enc,a_weights)])

    def output_vecs(self,states_enc,a_weights):
        return [vec * a_weight for vec, a_weight in zip(states_enc, a_weights)]


class FFAttention:
    """feed forward attention"""

    def __init__(self, model, dec_state_dim, enc_state_dim, att_dim):
        self.W1 = model.add_parameters((att_dim, enc_state_dim))
        self.W2 = model.add_parameters((att_dim, dec_state_dim))
        self.V = model.add_parameters((1, att_dim))

    def __call__(self, dec_state, enc_states):
        w1 = dy.parameter(self.W1)
        w2 = dy.parameter(self.W2)
        v = dy.parameter(self.V)

        attention_weights = []
        w2dt = w2 * dec_state
        for enc_state in enc_states:
            attention_weight = v * dy.tanh(w1 * enc_state + w2dt)
            attention_weights.append(attention_weight)
        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        return attention_weights

class FFAttention_Bernoulli:
    def __init__(self, model, dec_state_dim, enc_state_dim, att_dim):
        self.W1 = model.add_parameters((att_dim, enc_state_dim))
        self.W2 = model.add_parameters((att_dim, dec_state_dim))
        self.V = model.add_parameters((2, att_dim))
        self.transitions = model.add_lookup_parameters((2, 2))

    def compute_attention(self, dec_state, enc_states):
        w1 = dy.parameter(self.W1)
        w2 = dy.parameter(self.W2)
        v = dy.parameter(self.V)

        attention_potential = []
        w2dt = w2 * dec_state
        for enc_state in enc_states:
            logits = v * dy.tanh(w1 * enc_state + w2dt)
            attention_potential.append(logits)
        return attention_potential

    def __call__(self, dec_state, enc_states):
        attention_potential = self.compute_attention(dec_state, enc_states)
        attention_weights = self.forward_backward(attention_potential)
        return attention_weights

    def display(self, dec_state, enc_states, toks):
        attention_potential = self.compute_attention(dec_state, enc_states)
        best_path, _ = self.viterbi(attention_potential)
        output = []
        for tid, tag in enumerate(best_path):
            if tag == 1:
                output.append(toks[tid])
        return output

    @staticmethod
    def log_sum_exp(scores):
        npval = scores.npvalue()
        argmax_score = np.argmax(npval)
        max_score_expr = dy.pick(scores, argmax_score)
        max_score_expr_broadcast = dy.concatenate([max_score_expr] * 2)
        return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))

    def forward_backward(self, observations):
        init_alphas = [0, 0]
        forward_mess = dy.inputVector(init_alphas)
        alpha = []
        for i in range(len(observations) - 1):
            alphas_t = []
            for next_tag in range(2):
                obs_broadcast = dy.concatenate([dy.pick(observations[i], next_tag)] * 2)
                next_tag_expr = forward_mess + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(self.log_sum_exp(next_tag_expr))
            forward_mess = dy.concatenate(alphas_t)
            alpha.append(forward_mess)

        init_betas = [0, 0]
        backward_mess = dy.inputVector(init_betas)
        beta = []
        for i in range(len(observations) - 1):
            beta_t = []
            for next_tag in range(2):
                obs = observations[len(observations) - i - 1]
                next_tag_expr = backward_mess + self.transitions[next_tag] + obs
                beta_t.append(self.log_sum_exp(next_tag_expr))
            backward_mess = dy.concatenate(beta_t)
            beta.append(backward_mess)

        mu = [x + y for x, y in zip(alpha, beta[::-1])]
        # compute marginal probablities
        prob = [dy.pick(dy.softmax(w), 1) for w in mu]
        return prob

    def viterbi(self, observations):
        backpointers = []
        init_pis = [0, 0]
        forward_mess = dy.inputVector(init_pis)
        transitions = [self.transitions[idx] for idx in range(2)]
        for i in range(len(observations) - 1):
            bp_t = []
            pi_t = []
            for next_tag in range(2):
                next_tag_expr = forward_mess + transitions[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bp_t.append(best_tag_id)
                pi_t.append(dy.pick(next_tag_expr, best_tag_id))
            forward_mess = dy.concatenate(pi_t) + observations[i]
            backpointers.append(bp_t)
        # find the highrst scoring final state and the corresponding score
        best_tag_id = np.argmax(forward_mess.npvalue())
        path_score = dy.pick(forward_mess, best_tag_id)
        # backtracking
        best_path = [best_tag_id]
        for bp_t in reversed(backpointers):
            best_tag_id = bp_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.pop()
        best_path.reverse()
        return best_path, path_score

# a few generic functions to compute attention vector
def soft_average(buffer, word_weights):
    """soft attention"""
    return dy.esum([vector*attterion_weight for vector, attterion_weight in zip(buffer, word_weights)])

def hard_select(buffer, word_weights, wid):
    """hard attention when knowing where to attend"""
    return buffer[wid], word_weights[wid]

def hard_sample(buffer, word_weights, renorm_prob=0.8, restrict=None, argmax=False):
    """hard attention without knowing where to attend"""
    weights = word_weights.npvalue()
    if restrict != None:
        mask = np.zeros(len(buffer))
        mask[restrict] = 1
        weights = weights * mask
    renormed_weights = np.exp(weights * renorm_prob)
    renormed_weights = list(renormed_weights / renormed_weights.sum())
    if argmax:
        wid = max(enumerate(renormed_weights), key=itemgetter(1))[0]
    else:
        wid = np.random.choice(range(len(renormed_weights)), 1, p=renormed_weights)[0]
    return buffer[wid], wid

def threshold_sample(buffer, word_weights, renorm_prob=0.8, restrict=None, threshold=0.8):
    """hard attention that selects a subset of inputs whose value is greater than a threshold"""
    # threshold = 1.0 / len(buffer) * threshold
    weights = word_weights.npvalue()
    renormed_weights = np.exp(weights * renorm_prob)
    renormed_weights = list(renormed_weights / renormed_weights.sum())
    max_weight=max(renormed_weights)
    selected_vec = []
    selected_wid = []
    # wids=[]
    for wid, weight in enumerate(renormed_weights):
        if weight > threshold*max_weight:
            selected_vec.append(buffer[wid])
            selected_wid.append(word_weights[wid])
            # wids.append(wid)
    return dy.average(selected_vec), selected_wid

def attention_output(buffer, word_weights, attention_type, wid=None, renorm_prob=0.8, restrict=None, argmax=False):
    output_feature = None
    output_logprob = None
    wids=None

    if attention_type == 'soft_average':
        output_feature = soft_average(buffer, word_weights)
    elif attention_type == 'hard_select':
        output_feature, output_logprob = hard_select(buffer, word_weights, wid)
    elif attention_type == 'hard_sample':
        output_feature, wids = hard_sample(buffer, word_weights, renorm_prob=renorm_prob, restrict=restrict, argmax=argmax)
    elif attention_type == 'thre_sample':
        output_feature, wids = threshold_sample(buffer,word_weights,renorm_prob=renorm_prob,restrict=restrict,threshold=0.8)

    return output_feature, wids