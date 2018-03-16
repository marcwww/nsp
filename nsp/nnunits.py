import dynet as dy

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