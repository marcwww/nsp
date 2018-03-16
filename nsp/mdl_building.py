import dynet as dy

import ast
import config
import data
import mdl
import os


def train():

    optimizers = {
        "sgd": dy.SimpleSGDTrainer,
        "momentum": dy.MomentumSGDTrainer,
        "adam": dy.AdamTrainer,
        "adadelta": dy.AdadeltaTrainer,
        "adagrad": dy.AdagradTrainer
    }

    nnparser= mdl.NNParser(word_dict=data.word_dict,
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

    trainer=optimizers[config.opts.optimizer](nnparser.pc)

    idx=0
    for epoch in range(config.opts.epochs):
        sents = 0.0
        total_loss = 0.0
        train_size = len(data.word_tokens['train'])
        for word_tokens, act_tokens, term_tokens in data.\
                iter_data(data.word_tokens, data.act_tokens,
                          data.term_tokens, 'train'):

            # if(' '.join(x)=='how many states have a city called rochester'):
            #     print('a')
            loss = nnparser.train(word_tokens,act_tokens,term_tokens)
            sents += 1
            if loss is not None:
                total_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            e = float(idx) / train_size
            if idx % config.opts.print_every == 0:
                print('epoch {}: loss per sentence: {}'.format(e, total_loss / sents))
                sents = 0
                total_loss = 0.0

            idx += 1

        fname = 'epoch%03d.model' % (epoch)
        save_as = os.path.join(config.opts.model, fname)
        nnparser.save_mdl(save_as)