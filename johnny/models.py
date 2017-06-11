# import dill
# import tqdm
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
                 vocab_size,
                 pos_size,
                 pos_units=30,
                 word_units=100,
                 lstm_units=100,
                 num_lstm_layers=1,
                 dropout_inp=0.1,
                 dropout_rec=0.5,
                 mlp_arc_units=100,
                 mlp_lbl_units=100,
                 gpu_id=-1,
                 num_labels=46,
                 visualise = False
                 ):

        super(Dense, self).__init__()
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.pos_units = pos_units
        self.word_units = word_units
        self.lstm_units = lstm_units
        self.dropout_inp = dropout_inp
        self.dropout_rec = dropout_rec
        self.mlp_arc_units = mlp_arc_units
        self.mlp_lbl_units = mlp_lbl_units
        self.num_labels = num_labels
        self.gpu_id = gpu_id
        self.visualise = visualise
        if visualise:
            self.sleep_time = 0

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

        self.add_link('vT', L.Linear(mlp_arc_units, 1))
        self.add_link('U_arc', L.Linear(2*lstm_units, mlp_arc_units))
        self.add_link('W_arc', L.Linear(2*lstm_units, mlp_arc_units))

        self.add_link('V_lblT', L.Linear(mlp_lbl_units, self.num_labels))
        self.add_link('U_lbl', L.Linear(2*lstm_units, mlp_lbl_units))
        self.add_link('W_lbl', L.Linear(2*lstm_units, mlp_lbl_units))

    # def _create_batch(self, seq):
    #     max_seq_len = len(seq[0])
    #     seq = np.vstack([np.pad(np.array([sent[i] for sent in sents if i < len(sent)], dtype=np.int32)
    #                    for i in range(max_sent_len)])
    def _create_lstm_batch(self, seq):
        max_seq_len = len(seq[0])
        batch = self.xp.array([[sent[i] if i < len(sent)
                                else self.CHAINER_IGNORE_LABEL
                                for sent in seq]
                               for i in range(max_seq_len)],
                              dtype=np.int32)
        # TODO - only if gpu mode
        if self.gpu_id >= 0:
            cuda.to_gpu(batch, self.gpu_id)
        return batch




    def _feed_lstms(self, lstm_layers, sents, tags, boundaries):
        """pass batches of data through the lstm layers
        and store the activations"""
        # words and tags should have same length
        assert(len(sents) == len(tags))
        max_sent_len = len(sents)
        batch_size = len(sents[0])
        f_or_b = lstm_layers[-1][:1]
        state_name = '%s_lstm_states' % f_or_b
        # set f_lstm_states or b_lstm_states
        setattr(self, state_name, [])
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
            self[state_name].append(F.reshape(top_h, (batch_size, 1, self.lstm_units)))

    def __call__(self, s_words, s_pos, heads=None, labels=None):
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
        train = (heads is not None)
        # in order to process batches of different sized sentences using LSTM in chainer
        # we need to sort by sentence length.
        # The longest sentences in tokens need to be at the beginning of the
        # list, since chainer will simply not update the states corresponding
        # to the smallest sentences that have 'run out of tokens'.
        # We keep the permutation indices in order to reshuffle the output states,
        # since we want to map the activations to the inputs.
        if train:
            perm_indices, sorted_batch = zip(*sorted(enumerate(zip(s_words, s_pos, heads, labels)),
                                                     key=lambda x: len(x[1][0]),
                                                     reverse=True))
            f_sents, f_tags, sorted_heads, sorted_labels = zip(*sorted_batch)
            # print('heads ', sorted_heads)
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

        # mask batches for use in lstm
        f_sents = self._create_lstm_batch(f_sents)
        b_sents = self._create_lstm_batch(b_sents)
        f_tags = self._create_lstm_batch(f_tags)
        b_tags = self._create_lstm_batch(b_tags)

        if train:
            heads = self._create_lstm_batch(sorted_heads)
            labels = self._create_lstm_batch(sorted_labels)

        # mask needed because sentences aren't all the same length
        mask = (f_sents != self.CHAINER_IGNORE_LABEL).T
        col_lengths = np.sum(mask, axis=0)
        # feed lists of words into forward and backward lstms
        # each list is a column of words if we imagine the sentence of each batch
        # concatenated vertically
        self._feed_lstms(self.f_lstm, f_sents, f_tags, col_lengths)
        self._feed_lstms(self.b_lstm, b_sents, b_tags, col_lengths)
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
        minf = Variable(self.xp.full((batch_size, max_sent_len), self.MIN_PAD,
                                           dtype=self.xp.float32))
        corrected_align = []
        for i, l in enumerate(sent_lengths):
            # set what to throw away
            perm = np.hstack([   # reverse beginning of list
                              np.arange(l-1, -1, -1, dtype=np.int32),
                                 # leave rest of elements the same
                              np.arange(l, max_sent_len, dtype=np.int32)])
            correct = F.permutate(joint_b_states[i], perm, axis=0)
            corrected_align.append(F.reshape(correct, (1, max_sent_len, -1)))
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
        # reshape to 2D to calculate matrix multiplication
        collapsed_comb_states = F.reshape(comb_states, (-1, self.lstm_units * 2))
        u_arc = self.U_arc(collapsed_comb_states)
        u_arc = F.swapaxes(F.reshape(u_arc, (batch_size, -1, self.mlp_arc_units)), 0, 1)
        # we can also pre-calculate W * a_i , for all a_i
        # bs * max_sent x mlp_arc_units
        w_arc = self.W_arc(collapsed_comb_states)
        # max_sent x bs x mlp_arc_units
        w_arc = F.swapaxes(F.reshape(w_arc, (batch_size, -1, self.mlp_arc_units)), 0, 1)

        u_lbl = self.U_lbl(collapsed_comb_states)
        u_lbl = F.swapaxes(F.reshape(u_lbl, (batch_size, -1, self.mlp_lbl_units)), 0, 1)

        w_lbl = self.W_lbl(collapsed_comb_states)
        w_lbl = F.swapaxes(F.reshape(w_lbl, (batch_size, -1, self.mlp_lbl_units)), 0, 1)

        # the probability of each word being the head of sent[i]
        sent_arcs, arc_preds_wrong_order, lbl_preds_wrong_order = [], [], []
        self.loss = 0
        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            num_active = col_lengths[i]
            # if we are training - create label variable
            if train:
                # i-1 because sentence has root appended to beginning
                i_h = i-1
                # gold_heads = Variable(cuda.to_gpu(self.xp.array([sent[i_h] for sent in sorted_heads
                #                                 if i_h < len(sent)],
                #                                dtype=np.int32)))
                gold_heads = Variable(heads[i_h][:num_active])
                gold_labels = Variable(labels[i_h][:num_active])
            # We broadcast w_arc[i] to the size of u_as since we want to add
            # the activation of a_i to all different activations a_j
            a_u, a_w = F.broadcast(u_arc, w_arc[i])
            # compute U * a_j + V * a_i for all j and this loops i
            UWa = F.reshape(F.tanh(a_u + a_w), (-1, self.mlp_arc_units))
            # compute g(a_j, a_i)
            g_a = self.vT(UWa)
            arcs = F.swapaxes(F.reshape(g_a, (-1, batch_size)), 0, 1)
            arcs = F.where(mask, arcs, minf)[:num_active]
            if train:
                loss = F.softmax_cross_entropy(arcs, gold_heads,
                                               ignore_label=self.CHAINER_IGNORE_LABEL)
                self.loss += loss
                # print(self.loss.data)
            # can't append to predictions after padding
            # because we get elements we shouldn't
            # pred is the index of what we believe to be the head
            # ----------------------------------------
            # TODO: might need to be chainer function instead of np
            # ----------------------------------------
            arc_pred = np.argmax(arcs.data, 1)
            arc_preds_wrong_order.append(arc_pred)

            # print(arc_pred.shape)
            # print(arc_pred)
            l_heads = u_lbl[arc_pred, np.arange(len(arc_pred)), :]
            l_w = w_lbl[i][:num_active]
            # l_heads, l_w = F.broadcast(l_heads, w_lbl[i])
            # print(l_heads.shape)
            UWl = F.reshape(F.tanh(l_heads + l_w), (-1, self.mlp_lbl_units))
            lbls = self.V_lblT(UWl)
            if train:
                loss = F.softmax_cross_entropy(lbls, gold_labels,
                                               ignore_label=self.CHAINER_IGNORE_LABEL)
                self.loss += loss

            lbl_pred = np.argmax(lbls.data, 1)
            lbl_preds_wrong_order.append(lbl_pred)
            # we only bother actually getting the softmax values
            # if we are to visualise the results
            if self.visualise:
                # replace logits with prob from softmax
                arcs = F.softmax(arcs)
                arcs = F.pad(arcs, [(0, int(batch_size - num_active)), (0, 0)],
                             'constant', constant_values=np.exp(self.MIN_PAD))
                lbls = F.softmax(lbls)
                one_hot_index = np.zeros(max_sent_len, dtype=np.float32)
                one_hot_arc = np.zeros(max_sent_len, dtype=np.float32)
                correct_head_index = int(gold_heads.data[0])
                one_hot_arc[correct_head_index] = 1.
                one_hot_index[i] = 1.
                one_hot_lbl = np.zeros(self.num_labels, dtype=np.float32)
                correct_lbl_index = int(gold_labels.data[0]) 
                one_hot_lbl[correct_lbl_index] = 1.
                print(discrete_print('\n\nCur index : %-110s\nReal head : %-110s\nPred head : %-110s\n\n'
                                     'Real label: %-110s\nPred label: %-110s\n\n'
                                     'Sleep time: %.2f - change with up and down arrow keys') % (
                     '[%s] %d' % (bar(one_hot_index[:90]), i),
                     '[%s] %d |%d|' % (bar(one_hot_arc[:90]), correct_head_index, abs(correct_head_index - i)),
                        '[%s]' % bar(arcs.data[0].reshape(-1)[:90]),
                        '[%s] %s' % (bar(one_hot_lbl), UDepVocab.TAGS[correct_lbl_index]),
                        '[%s]' % bar(lbls.data[0].reshape(-1)),
                        self.sleep_time),
                      end='', flush=True)
                sleep(self.sleep_time)
            else:
                arcs = F.pad(arcs, [(0, int(batch_size - num_active)), (0, 0)],
                             'constant', constant_values=self.MIN_PAD)
            # permute back to correct batch order
            arcs = F.permutate(arcs, np.array(perm_indices, dtype=np.int32))
            sent_arcs.append(F.reshape(arcs, (batch_size, -1, 1)))
        self.arcs = F.concat(sent_arcs, axis=2)
        # print(self.arcs.shape)
        # print(self.arcs.data)
        # inverse permutation indices - to undo the permutation
        inv_perm_indices = [perm_indices.index(i) for i in range(len(perm_indices))]
        arc_preds = [[pred[i] for pred in arc_preds_wrong_order
                      if i < len(pred)]
                     for i in inv_perm_indices] 
        lbl_preds = [[pred[i] for pred in lbl_preds_wrong_order
                      if i < len(pred)]
                     for i in inv_perm_indices] 
        return arc_preds, lbl_preds
        # TODO Think of ways of avoiding self prediction

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.f_lstm + self.b_lstm:
            self[lstm_name].reset_state()
