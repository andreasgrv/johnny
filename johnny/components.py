from itertools import chain
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


class SubwordEmbedder(chainer.Chain):

    def __init__(self, word_encoder, in_sizes=None, out_sizes=None, dropout=0.2):
        self.in_sizes = in_sizes or []
        self.out_sizes = out_sizes or []
        super(SubwordEmbedder, self).__init__()
        with self.init_scope():
            self.word_encoder = word_encoder
            for index, (in_size, out_size) in enumerate(zip(self.in_sizes, self.out_sizes), 1):
                embed_layer = L.EmbedID(in_size, out_size, ignore_label=-1)
                self.set_embed(index, embed_layer)
        self.dropout = dropout
        self.out_size = self.word_encoder.out_size
        if out_sizes:
            self.out_size += sum(out_sizes)
        self.is_subword = True

    def get_embed(self, index):
        return self['embed_%d' % index] if index > 0 else self.word_encoder

    def set_embed(self, index, embed):
        setattr(self, 'embed_%d' % index, embed)

    def __call__(self, *seqs):
        act = F.concat((self.get_embed(i)(s) for i, s in enumerate(seqs)), axis=1)
        if self.dropout > 0.:
            act = F.dropout(act, ratio=self.dropout)
        return act


class Encoder(chainer.Chain):
    """Encodes a sentence word by word using a recurrent neural network.
    Is also responsible for converting inputs to arrays and sending copying
    them to the gpu if gpu_id > 0"""

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

    def pad_batch(self, seqs):
        """Pads list of sequences of different length to max
        seq length - we can't send a list of sequences of
        different size to the gpu.
        
        At the same time as padding converts rows to columns.
        NOTE: seqs must be already sorted from longest to
        shortest (longest at 0 index) if feeding into lstm.
        Example:

        [[1,2,3], [4,5], [6]] -> [[1,4,6],[2,5,-1],[3,-1,-1]]
        """
        # NOTE: This function only encodes up to what is considered to be
        # the maximum sequence length of the encoder [the 0 index sequence].
        batch = self.xp.array([[sent[i] if i < len(sent)
                                else self.CHAINER_IGNORE_LABEL
                                for sent in seqs]
                               for i in range(self.max_seq_len)],
                              dtype=self.xp.int32)
        return batch

    def _feed_lstms(self, lstm_layers, *seqs):
        """Pass batches of data through the lstm layers
        and generate the sentence embeddings for each
        sentence in the batch"""
        # we will reshape top lstm layer output to below shape
        # for easy concatenation along the sequence axis - index 1
        h_vec_shape = (self.batch_size, 1, self.lstm_units)
        states = []
        for i in range(self.max_seq_len):
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

        sents = in_seqs[0]
        # all sequences in in_seqs are assumed to have length corresponding
        # to in_seqs[0]
        seq_lengths = [len(sent) for sent in sents]
        self.batch_size = len(seq_lengths)
        self.max_seq_len = seq_lengths[0]

        # before we modify input we check if our embedder handles subword
        # information. if so we want to pass the list of individual words
        # to the embedder for efficiency purposes (we can encode each word
        # and employ a lookup table on the actual forward pass for each
        # batch). We assume the subword tokens are passed in a 3D array
        # as in_seqs[0] -> sentences x words in sentence x tokens in words
        if getattr(self.embedder, 'is_subword', False):
            # we need to modify in_seqs
            in_seqs = list(in_seqs)
            # unique list of words sorted from longest to shortest
            word_set = set(chain.from_iterable(sents))
            word_encoder = self.embedder.word_encoder
            word_encoder.encode_words(word_set)
            # replace 3D input with 2D - words replaced with hash
            in_seqs[0] = tuple(tuple(map(word_encoder.word_to_index, s)) for s in sents)

        # f_sents is sentence_length x batch_size - f_sents[0] contains
        # the first token from all sentences in the batch
        fwd = [self.pad_batch(seq) for seq in in_seqs]

        if self.use_bilstm:
            # backward - also create the sentence in reverse order for the bilstm
            bwd = [self.pad_batch([sent[::-1] for sent in seq])
                   for seq in in_seqs]

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
            for i, l in enumerate(seq_lengths):
                # set what to throw away
                perm = self.xp.hstack([   # reverse beginning of list
                                       self.xp.arange(l-1, -1, -1, dtype=self.xp.int32),
                                          # leave rest of elements the same
                                       self.xp.arange(l, self.max_seq_len, dtype=self.xp.int32)])
                correct = F.permutate(joint_b_states[i], perm, axis=0)
                corrected_align.append(F.reshape(correct, (1, self.max_seq_len, -1)))
            # concatenate the batches again
            joint_b_states = F.concat(corrected_align, axis=0)

            comb_states = F.concat((joint_f_states, joint_b_states), axis=2)
        else:
            comb_states = joint_f_states
        if getattr(self.embedder, 'is_subword', False):
            # Remember to clear cache
            self.embedder.word_encoder.clear_cache()
        # comb_states =  F.swapaxes(comb_states, 0, 1)
        # returns batch_size x max_seq_len x num_units
        return comb_states

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.f_lstm + self.b_lstm:
            self[lstm_name].reset_state()


class SubwordEncoder(chainer.Chain):

    def __init__(self, vocab_size, num_units, num_layers,
                 inp_dropout=0.2, rec_dropout=0.2, use_bilstm=True):

        super(SubwordEncoder, self).__init__()
        with self.init_scope():
            self.embed_layer = L.EmbedID(vocab_size, num_units)
            # Forward
            self.f_lstm = ['f_lstm_%d' % i for i in range(num_layers)]
            for lstm_name in self.f_lstm:
                setattr(self, lstm_name, L.LSTM(num_units, num_units))
            # Backward
            if use_bilstm:
                self.b_lstm = ['b_lstm_%d' % i for i in range(num_layers)]
                for lstm_name in self.b_lstm:
                    setattr(self, lstm_name, L.LSTM(num_units, num_units))
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.inp_dropout = inp_dropout
        self.rec_dropout = rec_dropout
        self.use_bilstm = use_bilstm
        # self.add_link('out_fwd', L.Linear(num_units, num_units))
        # self.add_link('out_bwd', L.Linear(num_units, num_units))
        self.out_size = num_units * 2 if self.use_bilstm else num_units
        self.cache = dict()

    def __call__(self, batch):
        return self.embedding[batch.data]
        # return F.vstack([self.cache[word] for word in batch.data])

    def encode_words(self, word_list):
        # reset state at each batch
        self.reset_state()

        # sort words - longest first
        sorted_wl = sorted(word_list, key=lambda x: len(x), reverse=True)
        max_word_len = len(sorted_wl[0])

        if self.use_bilstm:
            rev_wl = [word[::-1] for word in sorted_wl]

        for i in range(max_word_len):
            c_v = self.xp.array([word[i] for word in sorted_wl if i < len(word)], dtype=self.xp.int32)
            # c_v = chainer.cuda.to_gpu(c_v)
            c = chainer.Variable(c_v)
            self.encode(c, self.f_lstm)
            if self.use_bilstm:
                rev_c_v = self.xp.array([word[i] for word in rev_wl if i < len(word)], dtype=self.xp.int32)
                # rev_c_v = chainer.cuda.to_gpu(rev_c_v)
                rev_c = chainer.Variable(rev_c_v)
                self.encode(rev_c, self.b_lstm)

        # concatenate forward and backward encoding
        if self.use_bilstm:
            self.embedding = F.concat((self[self.f_lstm[-1]].h, self[self.b_lstm[-1]].h))
        else:
            self.embedding = self[self.f_lstm[-1]].h
        for i, word in enumerate(sorted_wl):
            self.cache[tuple(word)] = i# F.reshape(embedding[i], shape=(1, -1))
    #
    # def hash_word(self, word):
    #     """We want to avoid having to pad subword tokens on the input,
    #     so instead we replace the word with a hash of the subword tokens.
    #     We only pass the subword tokens to the encode_words function as a
    #     2D list : words x subword tokens. This way we don't have to worry
    #     about 3D input to the sentence encoder at a higher level
    #     (sentences x words x subword tokens -> sentences x word_hashes)
    #     """
    #     # we assume words are some sort of iterable of subword tokens.
    #     # if they are not tuples, we convert to tuple for hashing.
    #     return hash(word) if isinstance(word, tuple) else hash(tuple(word))

    def word_to_index(self, word):
        return self.cache[tuple(word)]

    def encode(self, char, lstm_layer_list):
        # get embedding + dropout
        act = self.embed_layer(char)
        if self.inp_dropout > 0.:
            act = F.dropout(act, ratio=self.inp_dropout)

        # feed into lstm layers
        for lstm_layer in lstm_layer_list:
            act = self[lstm_layer](act)
            if self.rec_dropout > 0.:
                act = F.dropout(act, ratio=self.rec_dropout)

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.f_lstm + self.b_lstm:
            self[lstm_name].reset_state()

    def clear_cache(self):
        self.cache = dict()
