import six
import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer
from time import sleep
from chainer import Variable, cuda
from johnny.utils import bar, discrete_print
from johnny.dep import UDepVocab
from johnny.extern import DependencyDecoder


# TODO Check multiple roots issue
# TODO Reimplement visualisation
# TODO Add dropout to head and label MLPs
# TODO Check constraint algorithms + optimise
# TODO Maybe switch to using predicted arcs towards end of training
# TODO Think of ways of avoiding self prediction
class Dense(chainer.Chain):

    MIN_PAD = -100.
    TREE_OPTS = ['none', 'chu', 'eisner']

    def __init__(self,
                 encoder,
                 num_labels=46,
                 mlp_arc_units=100,
                 mlp_lbl_units=100,
                 treeify='chu',
                 gpu_id=-1,
                 visualise=False,
                 debug=False
                 ):

        super(Dense, self).__init__()
        self.num_labels = num_labels
        self.mlp_arc_units = mlp_arc_units
        self.mlp_lbl_units = mlp_lbl_units
        self.treeify = treeify.lower()
        self.gpu_id = gpu_id
        self.visualise = visualise
        self.debug = debug
        self.sleep_time = 0

        assert(treeify in self.TREE_OPTS)
        self.unit_mult = 2 if encoder.use_bilstm else 1

        with self.init_scope():
            self.encoder = encoder

            self.vT = L.Linear(mlp_arc_units, 1)
            # head
            self.H_arc = L.Linear(self.unit_mult*self.encoder.lstm_units, mlp_arc_units)
            # dependent
            self.D_arc = L.Linear(self.unit_mult*self.encoder.lstm_units, mlp_arc_units)

            self.V_lblT = L.Linear(mlp_lbl_units, self.num_labels)
            self.U_lbl = L.Linear(self.unit_mult*self.encoder.lstm_units, mlp_lbl_units)
            self.W_lbl = L.Linear(self.unit_mult*self.encoder.lstm_units, mlp_lbl_units)

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
            labels = self.encoder._pad_batch(sorted_labels)

        u_lbl = self.U_lbl(sent_states)
        u_lbl = F.swapaxes(F.reshape(u_lbl, (batch_size, -1, self.mlp_lbl_units)), 0, 1)

        w_lbl = self.W_lbl(sent_states)
        w_lbl = F.swapaxes(F.reshape(w_lbl, (batch_size, -1, self.mlp_lbl_units)), 0, 1)

        sent_lbls = []
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
            arc_pred = pred_heads[i-1]

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

    def __call__(self, *inputs, **kwargs):
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
        assert(len(inputs) >= 1)
        heads = kwargs.get('heads', None)
        labels = kwargs.get('labels', None)

        calc_loss = ((heads is not None) and (labels is not None))
        input_sent_lengths = [len(sent) - 1 for sent in inputs[0]]
        # in order to process batches of different sized sentences using LSTM in chainer
        # we need to sort by sentence length.
        # The longest sentences in tokens need to be at the beginning of the
        # list, since chainer will simply not update the states corresponding
        # to the smallest sentences that have 'run out of tokens'.
        # We keep the permutation indices in order to reshuffle the output states,
        # since we want to map the activations to the inputs.
        perm_indices, sorted_batch = zip(*sorted(enumerate(zip(*inputs)),
                                                 key=lambda x: len(x[1][0]),
                                                 reverse=True))
        sorted_inputs = list(zip(*sorted_batch))
        if calc_loss:
            sorted_heads = [heads[i] for i in perm_indices]
            sorted_labels = [labels[i] for i in perm_indices]
        else:
            sorted_heads, sorted_labels = None, None

        f_sents = sorted_inputs[0]
        batch_size = len(f_sents)
        max_sent_len = len(f_sents[0])

        # comb states is batch_size x sentence_length x lstm_units
        comb_states = self.encoder(*sorted_inputs)
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
                (-1, self.encoder.lstm_units * self.unit_mult))

        self.loss = 0

        if calc_loss:
            sorted_heads = self.encoder._pad_batch(sorted_heads)

        batch_stats = (batch_size, max_sent_len, self.encoder.col_lengths)

        arcs = self._predict_heads(comb_states_2d, self.encoder.mask, batch_stats,
                sorted_heads=sorted_heads)

        if self.debug:
            self.arcs = cuda.to_cpu(arcs.data)

        if self.treeify != 'none':
            # TODO: check multiple roots issue
            # We process the head scores to apply tree constraints
            arcs = cuda.to_cpu(arcs.data)
            # DependencyDecoder expects a square matrix - fill root col with zeros
            pd_arcs = np.pad(arcs, ((0, 0), (0, 0), (1, 0)), 'constant')
            dd = DependencyDecoder()
            if self.treeify == 'chu':
                # Just remove cycles, non-projective trees are ok
                arc_preds = np.array([dd.parse_nonproj(each)[1:] for each in pd_arcs])
            elif self.treeify == 'eisner':
                # Remove cycles and make sure trees are projective
                arc_preds = np.array([dd.parse_proj(each)[1:] for each in pd_arcs])
            else:
                raise ValueError('Unexpected method')
            p_arcs = self.encoder._pad_batch(arc_preds)
        else:
            # We ignore tree constraints - head predictions may create cycles
            # we pass predict_labels the gpu object
            arcs = arcs.data
            p_arcs = self.xp.argmax(arcs, axis=1)
            arc_preds = cuda.to_cpu(p_arcs)
            p_arcs = np.swapaxes(p_arcs, 0, 1)

        lbls = self._predict_labels(comb_states_2d, p_arcs, sorted_heads,
                batch_stats, sorted_labels=sorted_labels)

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

        lbls.data = cuda.to_cpu(lbls.data)

        inv_perm_indices = [perm_indices.index(i) for i in range(len(perm_indices))]
        if self.debug:
            self.arcs = self.arcs[inv_perm_indices]
        # permute back to correct batch order
        arcs = arc_preds[inv_perm_indices]
        lbls = lbls.data[inv_perm_indices]

        lbl_preds = np.argmax(lbls, axis=1)

        arc_preds = [arc_p[:l] for arc_p, l in zip(arcs, input_sent_lengths)]
        lbl_preds = [lbl_p[:l] for lbl_p, l in zip(lbl_preds, input_sent_lengths)]

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
        self.encoder.reset_state()
