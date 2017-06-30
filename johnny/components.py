import chainer
import chainer.functions as F
import chainer.links as L


class Embedder(chainer.Chain):
    """A general embedder that concatenates the embeddings of an arbitrary
    number of input sequences."""

    def __init__(self, in_sizes, out_sizes, dropout=0.2):
        """
        :in_sizes: list of ints specifying the input vocabulary size for each
        sequence.
        :out_sizes: list of ints specifying embedding size for each sequence.
        :dropout: float between 0 and 1, how much dropout to apply to the input.

        As an example, suppose we want to encode words and part of speech
        tags. If we want: word vocab -> 100 , word emb size -> 100,
        pos vocab -> 10, pos emb size -> 30, we would feed:

        in_sizes = (100, 10) and out_sizes = (100, 30)

        the __call__ method then assumes that you will feed in the
        sequences in the corresponding order - first word indices and then
        pos tag indices.
        """
        super(Embedder, self).__init__()
        assert(len(in_sizes) == len(out_sizes))
        with self.init_scope():
            for index, (in_size, out_size) in enumerate(zip(in_sizes, out_sizes)):
                embed_layer = L.EmbedID(in_size, out_size, ignore_label=-1)
                self.set_embed(index, embed_layer)
        self.dropout = dropout
        self.out_size = sum(out_sizes)

    def get_embed(self, index):
        return self['embed_%d' % index]

    def set_embed(self, index, embed):
        setattr(self, 'embed_%d' % index, embed)

    def __call__(self, *seqs):
        act = F.concat((self.get_embed(i)(s) for i, s in enumerate(seqs)), axis=1)
        if self.dropout > 0.:
            act = F.dropout(act, ratio=self.dropout)
        return act


class Encoder(chainer.Chain):
    """Encodes a sentence word by word using a recurrent neural network"""

    CHAINER_IGNORE_LABEL = -1
    MIN_PAD = -100.

    def __init__(self, embedder, use_bilstm=True, num_lstm_layers=1,
                 lstm_units=100, dropout=0.2, gpu_id=-1):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embedder = embedder
            self.f_lstm, self.b_lstm = [], []
            for i in range(num_lstm_layers):
                # if this is the first lstm layer we need to have same number of
                # units as pos_units + word_units - else we use lstm_units num units
                in_size = embedder.out_size if i == 0 else lstm_units
                f_name = 'f_lstm_%d' % i
                self.f_lstm.append(f_name)
                setattr(self, f_name, L.LSTM(in_size, lstm_units))
                if use_bilstm:
                    b_name = 'b_lstm_%d' % i
                    self.b_lstm.append(b_name)
                    setattr(self, b_name, L.LSTM(in_size, lstm_units))

        self.use_bilstm = use_bilstm
        self.num_lstm_layers = num_lstm_layers
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.gpu_id = gpu_id

        self.mask = None
        self.col_lengths = None

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
            chainer.cuda.to_gpu(batch, self.gpu_id)
        return batch

    def _feed_lstms(self, lstm_layers, *seqs):
        """Pass batches of data through the lstm layers
        and generate the sentence embeddings for each
        sentence in the batch"""
        sents = seqs[0]
        max_sent_len = len(sents)
        batch_size = len(sents[0])
        # we will reshape top lstm layer output to below shape
        # for easy concatenation along the sequence axis - index 1
        h_vec_shape = (batch_size, 1, self.lstm_units)
        states = []
        for i in range(max_sent_len):
            # only get embedding up to padding
            # needed to pad because otherwise can't move whole batch to gpu
            active_until = self.col_lengths[i]

            # embedder computes activation by embedding each input sequence and
            # concatenating the resulting vectors to a single vector per
            # index in the sentence. We don't embed the -1 ids since we only
            # process up to :active_until - this comes up because in a single
            # batch we may have different sentence lengths.
            act = self.embedder(*(chainer.Variable(seq[i][:active_until]) for seq in seqs))

            for layer in lstm_layers:
                act = self[layer](act)
                if self.dropout > 0:
                    act = F.dropout(act, ratio=self.dropout)
            top_h = self[lstm_layers[-1]].h
            # we reshape to allow easy concatenation of activations
            # along sequence dimension
            states.append(F.reshape(top_h, h_vec_shape))
        return F.concat(states, axis=1)


    def __call__(self, *in_seqs):
        """Creates lstm embedding of the sentence and pos tags.
        If use_bilstm is specified - the embedding is formed from
        concatenating the forward and backward activations
        corresponding to each word.
        """

        # all sequences in in_seqs are assumed to have length corresponding
        # to in_seqs[0]
        sent_lengths = [len(sent) for sent in in_seqs[0]]
        max_sent_len = sent_lengths[0]

        # f_sents is sentence_length x batch_size - f_sents[0] contains the first token from
        # all sentences in the batch
        fwd = [self._pad_batch(seq) for seq in in_seqs]

        if self.use_bilstm:
            # backward - also create the sentence in reverse order for the bilstm
            bwd = [self._pad_batch([sent[::-1] for sent in seq]) for seq in in_seqs]

        # mask batches for use in lstm
        # mask needed because sentences aren't all the same length
        # mask is batch_size x sentence_length
        self.mask = (fwd[0] != self.CHAINER_IGNORE_LABEL).T

        self.col_lengths = self.xp.sum(self.mask, axis=0)
        # total_tokens = self.xp.sum(col_lengths)
        # feed lists of words into forward and backward lstms
        # each list is a column of words if we imagine the sentence of each batch
        # concatenated vertically
        joint_f_states = self._feed_lstms(self.f_lstm, *fwd)
        if self.use_bilstm:
            joint_b_states = self._feed_lstms(self.b_lstm, *bwd)
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
        return comb_states

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.f_lstm + self.b_lstm:
            self[lstm_name].reset_state()
