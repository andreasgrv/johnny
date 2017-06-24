import six
import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer
from time import sleep
from chainer import Variable, cuda
from johnny.utils import bar, discrete_print
from johnny.dep import UDepVocab


class Dense(chainer.Chain):

    CHAINER_IGNORE_LABEL = -1
    MIN_PAD = -100.

    def __init__(self,
                 vocab_size=10000,
                 pos_size=30,
                 pos_units=30,
                 word_units=100,
                 lstm_units=100,
                 num_lstm_layers=1,
                 use_bilstm=True,
                 dropout_inp=0.1,
                 dropout_rec=0.5,
                 mlp_arc_units=100,
                 mlp_lbl_units=100,
                 gpu_id=-1,
                 num_labels=46,
                 visualise=False,
                 debug=False
                 ):

        super(Dense, self).__init__()
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.pos_units = pos_units
        self.word_units = word_units
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.use_bilstm = use_bilstm
        self.dropout_inp = dropout_inp
        self.dropout_rec = dropout_rec
        self.mlp_arc_units = mlp_arc_units
        self.mlp_lbl_units = mlp_lbl_units
        self.num_labels = num_labels
        self.gpu_id = gpu_id
        self.visualise = visualise
        self.debug = debug
        self.sleep_time = 0

        self.unit_mult = 2 if self.use_bilstm else 1

        self.add_link('embed_word', L.EmbedID(self.vocab_size, self.word_units,
            ignore_label=self.CHAINER_IGNORE_LABEL))
        self.add_link('embed_pos', L.EmbedID(self.pos_size, self.pos_units,
            ignore_label=self.CHAINER_IGNORE_LABEL))
        self.f_lstm, self.b_lstm = [], []
        for i in range(self.num_lstm_layers):
            # if this is the first lstm layer we need to have same number of
            # units as pos_units + word_units - else we use lstm_units num units
            in_size = pos_units + word_units if i == 0 else lstm_units
            f_name = 'f_lstm_%d' % i
            self.f_lstm.append(f_name)
            self.add_link(f_name, L.LSTM(in_size, lstm_units))
            if self.use_bilstm:
                b_name = 'b_lstm_%d' % i
                self.b_lstm.append(b_name)
                self.add_link(b_name, L.LSTM(in_size, lstm_units))

        self.add_link('vT', L.Linear(mlp_arc_units, 1))
        # head
        self.add_link('H_arc', L.Linear(self.unit_mult*lstm_units, mlp_arc_units))
        # dependent
        self.add_link('D_arc', L.Linear(self.unit_mult*lstm_units, mlp_arc_units))

        self.add_link('V_lblT', L.Linear(mlp_lbl_units, self.num_labels))
        self.add_link('U_lbl', L.Linear(self.unit_mult*lstm_units, mlp_lbl_units))
        self.add_link('W_lbl', L.Linear(self.unit_mult*lstm_units, mlp_lbl_units))

    def _pad_batch(self, seqs):
        """Pads list of sequences of different length to max
        seq length - we can't send a list of sequences of
        different size to the gpu.
        
        At the same time as padding converts rows to columns.
        NOTE: seqs must be already sorted from longest to
        shortest (longest at 0 index) if feeding into lstm.
        Example:

        [[1,2,3], [4,5], [6]] -> [[1,4,6],[2,5,-1],[3,-1,-1]]
        """
        max_seq_len = len(seqs[0])
        batch = self.xp.array([[sent[i] if i < len(sent)
                                else self.CHAINER_IGNORE_LABEL
                                for sent in seqs]
                               for i in range(max_seq_len)],
                              dtype=self.xp.int32)
        if self.gpu_id >= 0:
            cuda.to_gpu(batch, self.gpu_id)
        return batch

    def _feed_lstms(self, lstm_layers, sents, tags, boundaries):
        """Pass batches of data through the lstm layers
        and generate the sentence embeddings for each
        sentence in the batch"""
        # words and tags should have same length
        assert(len(sents) == len(tags))
        max_sent_len = len(sents)
        batch_size = len(sents[0])
        # we will reshape top lstm layer output to below shape
        # for easy concatenation along the sequence axis - index 1
        h_vec_shape = (batch_size, 1, self.lstm_units)
        states = []
        for i in range(max_sent_len):
            # only get embedding up to padding
            # needed to pad because otherwise can't move whole batch to gpu
            active_until = boundaries[i]

            words = Variable(sents[i][:active_until])
            pos = Variable(tags[i][:active_until])
            word_emb = self.embed_word(words)
            pos_emb = self.embed_pos(pos)
            act = F.concat((word_emb, pos_emb), axis=1)
            if self.dropout_inp > 0:
                act = F.dropout(act, ratio=self.dropout_inp)

            for layer in lstm_layers:
                act = self[layer](act)
                if self.dropout_rec > 0:
                    act = F.dropout(act, ratio=self.dropout_rec)
            top_h = self[lstm_layers[-1]].h
            # we reshape to allow easy concatenation of activations
            # along sequence dimension
            states.append(F.reshape(top_h, h_vec_shape))
        return F.concat(states, axis=1)


    def _encode_sents(self, f_sents, f_tags):
        """Creates lstm embedding of the sentence and pos tags.
        If use_bilstm is specified - the embedding is formed from
        concatenating the forward and backward activations
        corresponding to each word.
        """

        sent_lengths = [len(sent) for sent in f_sents]
        max_sent_len = sent_lengths[0]

        if self.use_bilstm:
            # also create the sentence in reverse order for the bilstm
            b_sents, b_tags = [sent[::-1] for sent in f_sents], [tags[::-1] for tags in f_tags]
            b_sents = self._pad_batch(b_sents)
            b_tags = self._pad_batch(b_tags)

        # mask batches for use in lstm
        # f_sents is sentence_length x batch_size - f_sents[0] contains the first token from
        # all sentences in the batch
        f_sents = self._pad_batch(f_sents)
        f_tags = self._pad_batch(f_tags)


        # mask needed because sentences aren't all the same length
        # mask is batch_size x sentence_length
        mask = (f_sents != self.CHAINER_IGNORE_LABEL).T

        col_lengths = self.xp.sum(mask, axis=0)
        # total_tokens = self.xp.sum(col_lengths)
        # feed lists of words into forward and backward lstms
        # each list is a column of words if we imagine the sentence of each batch
        # concatenated vertically
        joint_f_states = self._feed_lstms(self.f_lstm, f_sents, f_tags, col_lengths)
        if self.use_bilstm:
            joint_b_states = self._feed_lstms(self.b_lstm, b_sents, b_tags, col_lengths)
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
            corrected_align = []
            for i, l in enumerate(sent_lengths):
                # set what to throw away
                perm = self.xp.hstack([   # reverse beginning of list
                                       self.xp.arange(l-1, -1, -1, dtype=self.xp.int32),
                                          # leave rest of elements the same
                                       self.xp.arange(l, max_sent_len, dtype=self.xp.int32)])
                correct = F.permutate(joint_b_states[i], perm, axis=0)
                corrected_align.append(F.reshape(correct, (1, max_sent_len, -1)))
            # concatenate the batches again
            joint_b_states = F.concat(corrected_align, axis=0)

            comb_states = F.concat((joint_f_states, joint_b_states), axis=2)
        else:
            comb_states = joint_f_states
        return comb_states, mask, col_lengths

    def _predict_heads(self, sent_states, mask, batch_stats, sorted_heads=None):
        """For each token in the sentence predict which token in the sentence
        is its head."""

        batch_size, max_sent_len, col_lengths = batch_stats

        calc_loss = sorted_heads is not None

        # head activations for each token lstm activation
        h_arc = self.H_arc(sent_states)
        # we transform results to be indexable by sentence index for upcoming for loop
        # h_arc is now max_sent x bs x mlp_arc_units
        h_arc = F.swapaxes(F.reshape(h_arc, (batch_size, -1, self.mlp_arc_units)), 0, 1)
        # bs * max_sent x mlp_arc_units
        d_arc = self.D_arc(sent_states)
        # max_sent x bs x mlp_arc_units
        d_arc = F.swapaxes(F.reshape(d_arc, (batch_size, -1, self.mlp_arc_units)), 0, 1)

        # the values to use to mask softmax for head prediction
        # e ^ -100 is ~ zero (can be changed from self.MIN_PAD)
        mask_vals = Variable(self.xp.full((batch_size, max_sent_len),
                                           self.MIN_PAD,
                                           dtype=self.xp.float32))

        sent_arcs = []
        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            num_active = col_lengths[i]
            # if we are calculating loss create truth variables
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_heads = Variable(sorted_heads[i-1])
            # ================== HEAD PREDICTION ======================
            # Because some sentences may be shorter - only num_active of the
            # batch have valid activations for this token.
            # We need to replace the invalid ones with zeros - because
            # otherwise when broadcasting and summing we will modify valid
            # batch activations for earlier tokens of the sentence.
            invalid_pad = ((0, int(batch_size - num_active)), (0, 0))
            d_arc_pad = F.pad(d_arc[i][:num_active], invalid_pad, 'constant', constant_values=0.)
            # We broadcast w_arc[i] to the size of u_as since we want to add
            # the activation of a_i to all different activations a_j
            a_u, a_w = F.broadcast(h_arc, d_arc_pad)
            # compute U * a_j + V * a_i for all j and this loops i
            comb_arc = F.reshape(F.tanh(a_u + a_w), (-1, self.mlp_arc_units))
            # compute g(a_j, a_i)
            arc_logit = self.vT(comb_arc)
            arcs = F.swapaxes(F.reshape(arc_logit, (-1, batch_size)), 0, 1)
            arcs = F.where(mask, arcs, mask_vals)
            # Calculate losses
            if calc_loss:
                # we don't want to average out over seen words yet
                # NOTE: do not use ignore_label - in gpu mode gold_heads gets mutated
                head_loss = F.sum(F.softmax_cross_entropy(arcs[:num_active], gold_heads[:num_active], reduce='no'))
                self.loss += head_loss
            sent_arcs.append(F.reshape(arcs, (batch_size, -1, 1)))
        arcs = F.concat(sent_arcs, axis=2)
        return arcs

    def _predict_labels(self, sent_states, pred_heads, gold_heads, batch_stats,
                        sorted_labels=None):
        """Predict the label for each of the arcs predicted in _predict_heads."""
        batch_size, max_sent_len, col_lengths = batch_stats

        calc_loss = sorted_labels is not None
        if calc_loss:
            labels = self._pad_batch(sorted_labels)

        u_lbl = self.U_lbl(sent_states)
        u_lbl = F.swapaxes(F.reshape(u_lbl, (batch_size, -1, self.mlp_lbl_units)), 0, 1)

        w_lbl = self.W_lbl(sent_states)
        w_lbl = F.swapaxes(F.reshape(w_lbl, (batch_size, -1, self.mlp_lbl_units)), 0, 1)

        pred_heads = self.xp.rollaxis(pred_heads, 2)
        sent_lbls = []
        # ----------------------------------------
        # TODO: might need to be chainer function instead of np
        # ----------------------------------------
        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            # num_active
            num_active = col_lengths[i]
            # if we are calculating loss create truth variables
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_labels = Variable(labels[i-1])
                # might need actual variable here?
                true_heads = gold_heads[i-1]
            arc_pred = self.xp.argmax(pred_heads[i-1], 1)

            # ================== LABEL PREDICTION ======================
            # TODO: maybe we should use arc_pred sometimes in training??
            # NOTE: gold_heads values after num_active get mutated here
            # make sure you don't use ignore_label in softmax - even if ok in forward
            # it will be wrong in backprop (we limit to :self.num_active)
            # gh_copy = self.xp.copy(gold_heads.data)
            head_indices = true_heads if chainer.config.train else arc_pred

            l_heads = u_lbl[head_indices, self.xp.arange(len(head_indices)), :]
            l_w = w_lbl[i]
            UWl = F.reshape(F.tanh(l_heads + l_w), (-1, self.mlp_lbl_units))
            lbls = self.V_lblT(UWl)

            # Calculate losses
            if calc_loss:
                label_loss = F.sum(F.softmax_cross_entropy(lbls[:num_active], gold_labels[:num_active], reduce='no'))
                self.loss += label_loss
            sent_lbls.append(F.reshape(lbls, (batch_size, -1, 1)))
        lbls = F.concat(sent_lbls, axis=2)
        return lbls

    def __call__(self, s_words, s_pos, heads=None, labels=None):
        """ Expects a batch of sentences 
        so a list of K sentences where each sentence
        is a 2-tuple of indices of words, indices of pos tags.
        Example of 2 sentences:
            [([1,5,2], [4,7,1]), ([1,2], [3,4])]
              w1 w2      p1 p2
              |------- s1 -------|

        w = word, p = pos tag, s = sentence

        This is as slow as the longest sentence - so bucketing sentences
        of same size can speed up training - prediction.
        """
        calc_loss = ((heads is not None) and (labels is not None))
        input_sent_lengths = [len(sent) - 1 for sent in s_words]
        # in order to process batches of different sized sentences using LSTM in chainer
        # we need to sort by sentence length.
        # The longest sentences in tokens need to be at the beginning of the
        # list, since chainer will simply not update the states corresponding
        # to the smallest sentences that have 'run out of tokens'.
        # We keep the permutation indices in order to reshuffle the output states,
        # since we want to map the activations to the inputs.
        if calc_loss:
            perm_indices, sorted_batch = zip(*sorted(enumerate(zip(s_words, s_pos, heads, labels)),
                                                     key=lambda x: len(x[1][0]),
                                                     reverse=True))
            f_sents, f_tags, sorted_heads, sorted_labels = zip(*sorted_batch)
        else:
            perm_indices, sorted_batch = zip(*sorted(enumerate(zip(s_words, s_pos)),
                                                     key=lambda x: len(x[1][0]),
                                                     reverse=True))
            f_sents, f_tags = zip(*sorted_batch)
            sorted_heads = None
            sorted_labels = None
        assert(len(f_sents) == len(s_words))

        batch_size = len(f_sents)
        max_sent_len = len(f_sents[0])

        # comb states is batch_size x sentence_length x lstm_units
        comb_states, mask, col_lengths = self._encode_sents(f_sents, f_tags)
        # In order to predict which head is most probable for a given word
        # P(w_j == head | w_i, S)  -- note for our purposes w_j is the variable
        # we compute: g(a_j, a_i) = vT * tanh(U * a_j + V * a_i)
        # for all j and a particular i and then pass the resulting vector through
        # a softmax to get the probability distribution
        # ------------------------------------------------------
        # In g(a_j, a_i) we note that we can precompute the matrix multiplications
        # for each word, we consider all possible heads
        # we can pre-calculate U * a_j , for all a_j
        # reshape to 2D to calculate matrix multiplication
        comb_states_2d = F.reshape(comb_states,
                (-1, self.lstm_units * self.unit_mult))

        self.loss = 0

        if calc_loss:
            sorted_heads = self._pad_batch(sorted_heads)

        batch_stats = (batch_size, max_sent_len, col_lengths)

        arcs = self._predict_heads(comb_states_2d, mask, batch_stats,
                sorted_heads=sorted_heads)

        lbls = self._predict_labels(comb_states_2d, arcs.data, sorted_heads,
                batch_stats, sorted_labels=sorted_labels)

        # lbl_pred = self.xp.argmax(lbls.data, 1)
        # lbl_preds_wrong_order.append(lbl_pred)

        # we only bother actually getting the softmax values
        # if we are to visualise the results
        # if self.visualise:
            # replace logits with prob from softmax - we pad with the exp
            # of the MIN_PAD - since that would have been the value if we passed
            # the MIN_PAD through the softmax
            # arcs = F.softmax(arcs)
            # arcs = F.pad(arcs, [(0, int(batch_size - self.num_active)), (0, 0)],
            #              'constant', constant_values=self.xp.exp(self.MIN_PAD))
            # lbls = F.softmax(lbls)
            # lbls = F.pad(lbls, [(0, int(batch_size - self.num_active)), (0, 0)],
            #              'constant', constant_values=self.xp.exp(self.MIN_PAD))
            # self._visualise(i, arcs, lbls, gold_heads, gold_labels)
        # else:
            # arcs = F.pad(arcs, [(0, int(batch_size - self.num_active)), (0, 0)],
            #              'constant', constant_values=self.MIN_PAD)
            # lbls = F.pad(lbls, [(0, int(batch_size - self.num_active)), (0, 0)],
            #              'constant', constant_values=self.MIN_PAD)

        # normalize loss over all tokens seen
        # self.loss = self.loss / total_tokens

        arcs.data = cuda.to_cpu(arcs.data)
        lbls.data = cuda.to_cpu(lbls.data)

        # permute back to correct batch order
        inv_perm_indices = [perm_indices.index(i) for i in range(len(perm_indices))]
        arcs = arcs.data[inv_perm_indices]
        lbls = lbls.data[inv_perm_indices]
        if self.debug:
            self.arcs = arcs
        arc_preds = np.argmax(arcs, axis=1)
        lbl_preds = np.argmax(lbls, axis=1)

        arc_preds = [arc_p[:l] for arc_p, l in zip(arc_preds, input_sent_lengths)]
        lbl_preds = [lbl_p[:l] for lbl_p, l in zip(lbl_preds, input_sent_lengths)]

        # TODO Think of ways of avoiding self prediction
        return arc_preds, lbl_preds

    # def _visualise(self, i, arcs, lbls, gold_heads, gold_labels):
    #     max_sent_len = len(arcs[0])
    #     one_hot_index = self.xp.zeros(max_sent_len, dtype=self.xp.float32)
    #     one_hot_arc = self.xp.zeros(max_sent_len, dtype=self.xp.float32)
    #     correct_head_index = int(gold_heads.data[0])
    #     one_hot_arc[correct_head_index] = 1.
    #     one_hot_index[i] = 1.
    #     one_hot_lbl = self.xp.zeros(self.num_labels, dtype=self.xp.float32)
    #     correct_lbl_index = int(gold_labels.data[0]) 
    #     one_hot_lbl[correct_lbl_index] = 1.
    #     six.print_(discrete_print('\n\nCur index : %-110s\nReal head : %-110s\nPred head : %-110s\n\n'
    #                          'Real label: %-110s\nPred label: %-110s\n\n'
    #                          'Sleep time: %.2f - change with up and down arrow keys') % (
    #          '[%s] %d' % (bar(one_hot_index[:90]), i),
    #          '[%s] %d |%d|' % (bar(one_hot_arc[:90]), correct_head_index, abs(correct_head_index - i)),
    #             '[%s]' % bar(arcs.data[0].reshape(-1)[:90]),
    #             '[%s] %s' % (bar(one_hot_lbl), UDepVocab.TAGS[correct_lbl_index]),
    #             '[%s]' % bar(lbls.data[0].reshape(-1)),
    #             self.sleep_time),
    #           end='', flush=True)
    #     sleep(self.sleep_time)

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.f_lstm + self.b_lstm:
            self[lstm_name].reset_state()
