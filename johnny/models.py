import os
import re
import csv
import dill
import tqdm
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers, Variable


class DenseModel(object):

    def __init__(self,
                 vocab_size=1e4,
                 pos_size=16,
                 pos_units=30,
                 word_units=100,
                 lstm_units=100,
                 num_lstm_layers=2
                 ):

        super(DenseModel, self).__init__()
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.lstm_units = lstm_units

        self.add_link('embed_pos', L.EmbedID(pos_size, pos_units))
        self.add_link('embed_word', L.EmbedID(vocab_size, word_units))
        self.fwd_lstm, self.bwd_lstm = [], []
        self.num_lstm_layers = num_lstm_layers
        for i in range(self.num_lstm_layers):
            # if this is the first lstm layer we need to have same number of
            # units as pos_units + word_units - else we use lstm_units num units
            in_size = pos_size + vocab_size if i == 0 else lstm_units
            fwd_name = 'lstm_fwd_%d' % i
            bwd_name = 'lstm_bwd_%d' % i
            self.fwd_lstm.append(fwd_name)
            self.bwd_lstm.append(bwd_name)
            self.add_link(fwd_name, L.LSTM(in_size, lstm_units))
            self.add_link(bwd_name, L.LSTM(in_size, lstm_units))

        self.add_link('v', L.Linear(2*lstm_units, 1))
        self.add_link('U', L.Linear(2*lstm_units, 2*lstm_units))
        self.add_link('W', L.Linear(2*lstm_units, 2*lstm_units))

        def __call__(batch, train=False):
            """ Expects a batch of sentences 
            so a list of K sentences where each sentence
            is a 2-tuple of indices of words, indices of pos tags.
            Example of 2 sentences:
                [([1, 5, 2], [4, 7,1]), ([1,2], [3,4])]
                  w1 w2      p1 p2
                  |------- s1 -------|

            w = word, p = pos tag, s = sentence

            This is as slow as the longest sentence - so bucketing sentences
            of same size can speed up training - prediction.
            """
            # in order to process batches of different sized sentences using LSTM in chainer
            # we need to sort by sentence length.
            # The longest sentences in tokens need to be at the beginning of the
            # list, since chainer will simply not update the states corresponding
            # to the smallest sentences that have 'run out of tokens'.
            # We keep the permutation indices in order to reshuffle the output states,
            # since we want to map the activations to the inputs.
            perm_indices, sorted_batch = zip(*sorted(enumerate(batch),
                                                     key=lambda x: len(x[1][0]),
                                                     reverse=True))
            # words and pos should have same length
            words, pos = zip(*sorted_batch)
            max_sent_len = len(words[0])
            for i in range(max_sent_len):
                fwd_words = Variable(np.array([sent[i] for sent in words if i < len(sent)],
                                              dtype=np.int32),
                                     volatile=not train)
                fwd_pos = Variable(np.array([sent[i] for sent in pos if i < len(sent)],
                                            dtype=np.int32),
                                   volatile=not train)
                fwd_word_emb = self.embed_word(fwd_words)
                fwd_pos_emb = self.embed_pos(fwd_pos)
                fwd_act = F.concat(fwd_word_emb, fwd_pos_emb)

                for layer in self.fwd_lstm:
                    fwd_act = self[layer](fwd_act)
