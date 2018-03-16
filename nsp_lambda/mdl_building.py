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

    nnparser= mdl.NNParser(pretrained_embedding=None)

    trainer=optimizers[config.opts.optimizer](nnparser.pc)

    idx=0
    for epoch in range(config.opts.epochs):
        sents = 0.0
        total_loss = 0.0
        train_size = len(data.word_tokens['train.json'])
        for word_tokens, act_tokens, term_tokens in data.\
                iter_data(data.word_tokens, data.act_tokens,
                          data.term_tokens, 'train.json'):

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

        fname='epoch%03d.model' % (epoch)
        save_as=os.path.join(config.opts.model,fname)
        nnparser.save_mdl(save_as)

        # print('testing...')
        # fname='test_res%d' % (idx)
        # save_as=os.path.join(config.opts.result,fname)
        # rf = open(save_as, 'w')
        # test_sents = 0
        # test_loss = 0.0
        # for word_tokens, act_tokens, term_tokens in data. \
        #         iter_data(data.word_tokens, data.act_tokens,
        #                   data.term_tokens, 'test.json'):
        #     output_actions, output_tokens = \
        #         nnparser.parse(word_tokens)
        #
        #     raw_lf = mdl.recover_rlf(output_actions, output_tokens)
        #     rf.write(raw_lf + '\n')
        #
        # rf.close()