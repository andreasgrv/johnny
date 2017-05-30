# import dill
# import tqdm
import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer
from chainer import Variable


class Dense(chainer.Chain):

    CHAINER_IGNORE_LABEL = -1
    MIN_PAD = -100.

    def __init__(self,
                 vocab_size,
                 pos_size,
                 pos_units=30,
                 word_units=100,
                 lstm_units=100,
                 num_lstm_layers=2,
                 visualise = False
                 ):

        super(Dense, self).__init__()
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.pos_units = pos_units
        self.word_units = word_units
        self.lstm_units = lstm_units
        self.visualise = visualise

        self.add_link('embed_word', L.EmbedID(self.vocab_size, self.word_units))
        self.add_link('embed_pos', L.EmbedID(self.pos_size, self.pos_units))
        self.f_lstm, self.b_lstm = [], []
        self.num_lstm_layers = num_lstm_layers
        for i in range(self.num_lstm_layers):
            # if this is the first lstm layer we need to have same number of
            # units as pos_units + word_units - else we use lstm_units num units
            in_size = pos_units + word_units if i == 0 else lstm_units
            f_name = 'f_lstm_%d' % i
            b_name = 'b_lstm_%d' % i
            self.f_lstm.append(f_name)
            self.b_lstm.append(b_name)
            self.add_link(f_name, L.LSTM(in_size, lstm_units))
            self.add_link(b_name, L.LSTM(in_size, lstm_units))

        self.add_link('vT', L.Linear(2*lstm_units, 1))
        self.add_link('U', L.Linear(2*lstm_units, 2*lstm_units))
        self.add_link('W', L.Linear(2*lstm_units, 2*lstm_units))

    def _feed_lstms(self, lstm_layers, sents, tags, train):
        """pass batches of data through the lstm layers
        and store the activations"""
        assert(len(sents) == len(tags))
        batch_size = len(sents)
        # words and tags should have same length
        max_sent_len = len(sents[0])
        f_or_b = lstm_layers[-1][:1]
        state_name = '%s_lstm_states' % f_or_b
        # set f_lstm_states or b_lstm_states
        setattr(self, state_name, [])
        for i in range(max_sent_len):
            words = Variable(np.array([sent[i] for sent in sents if i < len(sent)],
                                      dtype=np.int32),
                             volatile=not train)
            pos = Variable(np.array([sent[i] for sent in tags if i < len(sent)],
                                    dtype=np.int32),
                           volatile=not train)
            word_emb = self.embed_word(words)
            pos_emb = self.embed_pos(pos)
            act = F.concat((word_emb, pos_emb), axis=1)

            for layer in lstm_layers:
                act = self[layer](act)
            top_h = self[lstm_layers[-1]].h
            # we reshape to allow easy concatenation of activations
            self[state_name].append(F.reshape(top_h, (batch_size, 1, self.lstm_units)))

    def __call__(self, s_words, s_pos, targets=None):
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
        # if we pass targets, we are training!
        train = (targets is not None)
        # in order to process batches of different sized sentences using LSTM in chainer
        # we need to sort by sentence length.
        # The longest sentences in tokens need to be at the beginning of the
        # list, since chainer will simply not update the states corresponding
        # to the smallest sentences that have 'run out of tokens'.
        # We keep the permutation indices in order to reshuffle the output states,
        # since we want to map the activations to the inputs.
        if train:
            perm_indices, sorted_batch = zip(*sorted(enumerate(zip(s_words, s_pos, targets)),
                                                     key=lambda x: len(x[1][0]),
                                                     reverse=True))
            f_sents, f_tags, sorted_targets = zip(*sorted_batch)
            # print('heads ', sorted_targets)
        else:
            perm_indices, sorted_batch = zip(*sorted(enumerate(zip(s_words, s_pos)),
                                                     key=lambda x: len(x[1][0]),
                                                     reverse=True))
            f_sents, f_tags = zip(*sorted_batch)
        assert(len(f_sents) == len(s_words))
        # print('f_sents ', f_sents)
        # print('f_tags ', f_tags)
        # also create the sentence in reverse order for the bilstm
        b_sents, b_tags = [sent[::-1] for sent in f_sents], [tags[::-1] for tags in f_tags]

        sent_lengths = [len(sent) for sent in f_sents]

        batch_size = len(f_sents)
        max_sent_len = len(f_sents[0])

        # feed lists of words into forward and backward lstms
        # each list is a column of words if we imagine the sentence of each batch
        # concatenated vertically
        self._feed_lstms(self.f_lstm, f_sents, f_tags, train)
        self._feed_lstms(self.b_lstm, b_sents, b_tags, train)
        joint_f_states = F.concat(self.f_lstm_states, axis=1)
        # need to shift activations because of sentence length difference
        # ------------------------- EXPLANATION -------------------------
        # assume rows are batches of sentences - and each number a vector
        # of numbers # - the activation that corresponds to the word, -1
        # represents the fact that the sentence is shorter.
        # ---------------------------------------------------------------
        # fwd lstm act :  [1,  2,  3]       bwd lstm act :  [3 ,  2,  1]
        #                 [4,  5, -1]                       [5 ,  4, -1]
        #                 [6, -1, -1]                       [6 , -1, -1]
        # 
        # Notice that we cant simply reverse the backword lstm activations
        # and concatenate them - because for smaller sentences we would be
        # concatenating with states that don't correspond to words.
        # So we reverse the activations up to the length of the sentence.
        # ---------------------------------------------------------------
        # fwd lstm act :  [1,  2,  3]       bwd lstm act :  [1 , 2,  3]
        #                 [4,  5, -1]                       [4 , 5, -1]
        #                 [6, -1, -1]                       [6, -1, -1]
        # ---------------------------------------------------------------
        joint_b_states = F.concat(self.b_lstm_states, axis=1)
        # mask needed because sentences aren't all the same length
        mask = self.xp.ones((batch_size, max_sent_len), dtype=np.bool)
        minf = Variable(self.xp.full((batch_size, max_sent_len), self.MIN_PAD,
                                           dtype=self.xp.float32), volatile=(not train))
        corrected_align = []
        for i, l in enumerate(sent_lengths):
            # set what to throw away
            mask[i, l:] = 0
            perm = np.hstack([   # reverse beginning of list
                              np.arange(l-1, -1, -1, dtype=np.int32),
                                 # leave rest of elements the same
                              np.arange(l, max_sent_len, dtype=np.int32)])
            correct = F.permutate(joint_b_states[i], perm, axis=0)
            corrected_align.append(F.reshape(correct, (1, max_sent_len, -1)))
        col_lengths = np.sum(mask, axis=0)
        # concatenate the batches again
        joint_b_states = F.concat(corrected_align, axis=0)

        comb_states = F.concat((joint_f_states, joint_b_states), axis=2)

        # In order to predict which head is most probable for a given word
        # P(w_j == head | w_i, S)  -- note for our purposes w_j is the variable
        # we compute: g(a_j, a_i) = vT * tanh(U * a_j + V * a_i)
        # for all j and a particular i and then pass the resulting vector through
        # a softmax to get the probability distribution
        # ------------------------------------------------------
        # In g(a_j, a_i) we note that we can precompute the matrix multiplications
        # for each word, we consider all possible heads
        # we can pre-calculate U * a_j , for all a_j
        u_as = self.U(F.reshape(comb_states, (-1, self.lstm_units * 2)))
        u_as = F.swapaxes(F.reshape(u_as, (batch_size, -1, self.lstm_units * 2)), 0, 1)
        # we can also pre-calculate W * a_i , for all a_i
        # bs * max_sent x units * 2
        w_as = self.W(F.reshape(comb_states, (-1, self.lstm_units * 2)))
        # max_sent x bs x units * 2
        w_as = F.swapaxes(F.reshape(w_as, (batch_size, -1, self.lstm_units * 2)), 0, 1)

        # the probability of each word being the head of sent[i]
        sent_attn, preds_wrong_order = [], []
        self.loss = 0
        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            num_active = col_lengths[i]
            # if we are training - create label variable
            if train:
                # i-1 because sentence has root appended to beginning
                i_h = i-1
                gold_heads = Variable(np.array([sent[i_h] for sent in sorted_targets
                                                if i_h < len(sent)],
                                               dtype=np.int32),
                                      volatile=False)
            # We broadcast w_as[i] to the size of u_as since we want to add
            # the activation of a_i to all different activations a_j
            a_u, a_v = F.broadcast(u_as, w_as[i])
            # compute U * a_j + V * a_i for all j and this loops i
            UWact = F.reshape(F.tanh(a_u + a_v), (-1, self.lstm_units * 2))
            # compute g(a_j, a_i)
            g_a = self.vT(UWact)
            attn = F.swapaxes(F.reshape(g_a, (-1, batch_size)), 0, 1)
            attn = F.where(mask, attn, minf)[:num_active]
            if train:
                loss = F.softmax_cross_entropy(attn, gold_heads,
                                               ignore_label=self.CHAINER_IGNORE_LABEL)
                self.loss += loss
                # print(self.loss.data)
            # can't append to predictions after padding
            # because we get elements we shouldn't
            preds_wrong_order.append(np.argmax(attn.data, 1))
            # we only bother actually getting the softmax values
            # if we are to visualise the results
            if self.visualise:
                # replace logits with prob from softmax
                attn = F.softmax(attn)
                attn = F.pad(attn, [(0, batch_size - num_active), (0, 0)],
                             'constant', constant_values=np.exp(self.MIN_PAD))
            else:
                attn = F.pad(attn, [(0, batch_size - num_active), (0, 0)],
                             'constant', constant_values=self.MIN_PAD)
            # permute back to correct batch order
            attn = F.permutate(attn, np.array(perm_indices, dtype=np.int32))
            sent_attn.append(F.reshape(attn, (batch_size, -1, 1)))
        self.attn = F.concat(sent_attn, axis=2)
        # print(self.attn.shape)
        # print(self.attn.data)
        # inverse permutation indices - to undo the permutation
        inv_perm_indices = [perm_indices.index(i) for i in range(len(perm_indices))]
        preds = [[pred[i] for pred in preds_wrong_order
                 if i < len(pred)]
                 for i in inv_perm_indices] 
        return preds
        # TODO Think of ways of avoiding self prediction

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.f_lstm + self.b_lstm:
            self[lstm_name].reset_state()
