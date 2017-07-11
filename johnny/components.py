from itertools import chain
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.links.connection.n_step_lstm as chainer_nstep
from johnny.extern import NStepLSTMBase

CHAINER_IGNORE_LABEL = -1

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
                embed_layer = L.EmbedID(in_size, out_size, ignore_label=CHAINER_IGNORE_LABEL)
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
                embed_layer = L.EmbedID(in_size, out_size, ignore_label=CHAINER_IGNORE_LABEL)
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
    """Encodes a sentence word by word using a recurrent neural network."""


    CHAINER_IGNORE_LABEL = -1

    def __init__(self, embedder, use_bilstm=True, num_layers=1,
                 num_units=100, dropout=0.2):

        super(Encoder, self).__init__()
        with self.init_scope():
            self.embedder = embedder
            # we already have sorted input, so we removed the code that permutes
            # input and output (hence why we don't use the chainer class)
            self.rnn = NStepLSTMBase(num_layers,
                                     self.embedder.out_size,
                                     num_units,
                                     dropout,
                                     use_bi_direction=use_bilstm)

        self.use_bilstm = use_bilstm
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout = dropout

        self.mask = None
        self.col_lengths = None

    def transpose_batch(self, seqs, create_var=True):
        """transpose list of sequences of different length

        NOTE: seqs must be already sorted from longest to
        shortest (longest at 0 index) if feeding into lstm.
        Example:

        [[1,2,3], [4,5], [6]] -> [[1,4,6],[2,5],[3]]
        """
        # NOTE: This function only encodes up to what is considered to be
        # the maximum sequence length of the encoder [the 0 index sequence].
        if create_var:
            batch = [chainer.Variable(self.xp.array([sent[i]
                                                    for sent in seqs
                                                    if i < len(sent)],
                                      dtype=self.xp.int32))
                     for i in range(self.max_seq_len)]
        else:
            batch = [self.xp.array([sent[i]
                                    for sent in seqs
                                    if i < len(sent)],
                                   dtype=self.xp.int32)
                     for i in range(self.max_seq_len)]
        return batch


    def __call__(self, *in_seqs):
        """Creates lstm embedding of the sentence and pos tags.
        If use_bilstm is specified - the embedding is formed from
        concatenating the forward and backward activations
        corresponding to each word.
        """
        sents = in_seqs[0]
        # all sequences in in_seqs are assumed to have length corresponding
        # to in_seqs[0]
        self.max_seq_len = len(sents[0])
        self.batch_size = len(sents)

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

        # turn batch_size x seq_len -> seq_len x batch_size
        # NOTE: seq_len is variable - we aren't padding
        fwd = [self.transpose_batch(seq) for seq in in_seqs]

        # collapse all ids into a vector
        embeddings = self.embedder(*(F.concat(f, axis=0) for f in fwd))

        self.col_lengths = [len(col) for col in fwd[0]]

        # use np because cumsum crashes gpu - I know, right?
        self.batch_split = np.cumsum(self.col_lengths[:-1])
        # split back to batch size
        batch_embeddings = F.split_axis(embeddings, self.batch_split, axis=0)

        _, _, states = self.rnn(None, None, batch_embeddings)

        states = F.vstack((F.pad(s,
                                 ((0, self.batch_size - len(s)), (0,0)),
                                 'constant',
                                 constant_values=0.)
                           for s in states))

        mask_shape = (self.batch_size, self.max_seq_len)
        self.mask = self.xp.ones(mask_shape, dtype=self.xp.bool_)
        for i, l in enumerate(self.col_lengths):
            self.mask[l:, i] = False

        if getattr(self.embedder, 'is_subword', False):
            # Remember to clear cache
            self.embedder.word_encoder.clear_cache()

        # encoding of tokens at i padded to batch_size
        # returns max_seq_len x batch_size x num_units
        return states


class SubwordEncoder(chainer.Chain):

    def __init__(self, vocab_size, num_units, num_layers,
                 inp_dropout=0.2, rec_dropout=0.2, use_bilstm=True):

        super(SubwordEncoder, self).__init__()
        with self.init_scope():
            self.embed_layer = L.EmbedID(vocab_size, num_units,
                                         ignore_label=CHAINER_IGNORE_LABEL)
            self.rnn = chainer_nstep.NStepLSTMBase(num_layers,
                                                   num_units,
                                                   num_units,
                                                   rec_dropout,
                                                   use_bi_direction=use_bilstm)
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

    def encode_words(self, word_list):

        word_lengths = [len(w) for w in word_list]
        batch_split = np.cumsum(word_lengths[:-1])

        word_vars = [chainer.Variable(self.xp.array(w, dtype=self.xp.int32))
                                      for w in word_list]
        embeddings = self.embed_layer(F.concat(word_vars, axis=0))

        # split back to batch size
        batch_embeddings = F.split_axis(embeddings, batch_split, axis=0)
        _, _, hs = self.rnn(None, None, batch_embeddings)
        self.embedding = F.vstack([h[-1] for h in hs])

        for i, word in enumerate(word_list):
            self.cache[tuple(word)] = i

    def word_to_index(self, word):
        return self.cache[tuple(word)]

    def clear_cache(self):
        self.cache = dict()


class CNNSubwordEncoder(chainer.Chain):

    FILTER_MULTIPLIER = 25
    IGNORE_LABEL = -1

    def __init__(self, vocab_size, embed_units=15, num_highway_layers=1,
                 inp_dropout=0.2, ngrams=(1, 2, 3, 4, 5, 6), stride=1, num_filters=None):

        super(CNNSubwordEncoder, self).__init__()
        if num_filters is None:
            # http://www.people.fas.harvard.edu/~yoonkim/data/char-nlm.pdf
            # Table 2 small model uses constant size
            num_filters = [n * self.FILTER_MULTIPLIER for n in ngrams]
        assert(len(num_filters) == len(ngrams))
        assert(num_highway_layers >= 0)
        out_size = sum(num_filters)
        with self.init_scope():
            self.embed_layer = L.EmbedID(vocab_size, embed_units,
                                         ignore_label=self.IGNORE_LABEL)
            self.cnn_blocks = ['cnn_%d' % n for n in ngrams]
            self.highways = ['highway_%d' % i for i in range(num_highway_layers)]
            # for n in ngrams:
            #     setattr(self, self.cnn_blocks[n])
            for i, name in enumerate(self.cnn_blocks):
                setattr(self, name, L.Convolution2D(1,
                                       num_filters[i],
                                       (ngrams[i], embed_units),
                                       stride))
            for name in self.highways:
                setattr(self, name, L.Highway(out_size))
        self.vocab_size = vocab_size
        self.embed_units = embed_units
        self.num_highway_layers = num_highway_layers
        self.inp_dropout = inp_dropout
        self.out_size = out_size
        self.cache = dict()

    def __call__(self, batch):
        return self.embedding[batch.data]

    def encode_words(self, word_list):

        batch_size = len(word_list)
        sorted_word_list = sorted(word_list, key=lambda x: len(x), reverse=True)

        # split into parts - because there will be very few very long words
        # might as well pack them together to avoid wasting computation on padding
        if batch_size > 40:
            SPLIT_INDEX = int(0.05 * batch_size)
            batch_split = np.array_split(sorted_word_list, [SPLIT_INDEX])
        else:
            batch_split = [sorted_word_list]

        acts = None
        for shard in batch_split:

            shard_len = len(shard)
            max_shard_word_length = len(shard[0])
            word_vars = F.concat([chainer.Variable(self.xp.array([w[i]
                                                   if i < len(w)
                                                   else CHAINER_IGNORE_LABEL
                                                   for i in range(max_shard_word_length)],
                                                   dtype=self.xp.int32))
                                          for w in shard], axis=0)

            embeddings = self.embed_layer(word_vars)

            stacked = F.reshape(embeddings, (shard_len, -1, self.embed_units))

            stacked = F.expand_dims(stacked, axis=1)
            # for each in batch_embeddings:
            hs = None
            for block in self.cnn_blocks:
                h = self[block](stacked)
                h = F.max_pooling_2d(h, (max_shard_word_length, self.embed_units))
                # print(max_word_length, h.shape)
                h = F.squeeze(h)
                if hs is None:
                    hs = h
                else:
                    hs = F.concat([hs, h], axis=1)

            act = F.tanh(hs)
            # Don't apply dropout to last layer
            for highway in self.highways:
                if self.inp_dropout > 0.:
                    act = F.dropout(act, ratio=self.inp_dropout)
                act = self[highway](act)
            if acts is None:
                acts = act
            else:
                acts = F.vstack([acts, act])

        self.embedding = acts

        for i, word in enumerate(sorted_word_list):
            self.cache[tuple(word)] = i

    def word_to_index(self, word):
        return self.cache[tuple(word)]

    def clear_cache(self):
        self.cache = dict()


# class CNNDSubwordEncoder(chainer.Chain):
#
#     FILTER_MULTIPLIER = 25
#     def __init__(self, vocab_size, embed_units=15, num_highway_layers=1,
#                  inp_dropout=0.2, ngrams=(1, 2, 3, 4, 5, 6), stride=1, num_filters=None):
#
#         super(CNNDSubwordEncoder, self).__init__()
#         if num_filters is None:
#             # http://www.people.fas.harvard.edu/~yoonkim/data/char-nlm.pdf
#             # Table 2 small model uses constant size
#             num_filters = [n * self.FILTER_MULTIPLIER for n in ngrams]
#         assert(len(num_filters) == len(ngrams))
#         assert(num_highway_layers >= 0)
#         out_size = sum(num_filters)
#         with self.init_scope():
#             self.embed_layer = L.EmbedID(vocab_size, embed_units, ignore_label=CHAINER_IGNORE_LABEL)
#             self.cnn_blocks = ['cnn_%d' % n for n in ngrams]
#             self.highways = ['highway_%d' % i for i in range(num_highway_layers)]
#             # for n in ngrams:
#             #     setattr(self, self.cnn_blocks[n])
#             for i, name in enumerate(self.cnn_blocks):
#                 setattr(self, name, L.ConvolutionND(1, embed_units,
#                                        num_filters[i],
#                                        ngrams[i],
#                                        stride))
#             for name in self.highways:
#                 setattr(self, name, L.Highway(out_size))
#         self.vocab_size = vocab_size
#         self.embed_units = embed_units
#         self.num_highway_layers = num_highway_layers
#         self.inp_dropout = inp_dropout
#         self.out_size = out_size
#         self.cache = dict()
#
#     def __call__(self, batch):
#         return self.embedding[batch.data]
#
#     def encode_words(self, word_list):
#
#         batch_size = len(word_list)
#         word_lengths = [len(w) for w in word_list]
#         max_word_length = max(word_lengths)
#         batch_split = np.cumsum(word_lengths[:-1])
#
#         word_vars = [chainer.Variable(self.xp.array(w, dtype=self.xp.int32))
#                                       for w in word_list]
#
#         embeddings = self.embed_layer(F.concat(word_vars, axis=0))
#
#         batch_embeddings = F.split_axis(embeddings, batch_split, axis=0)
#
#         padded_batch_embeddings = F.pad_sequence(batch_embeddings, max_word_length)
#
#         stacked = F.stack(padded_batch_embeddings, axis=0)
#
#         stacked = F.swapaxes(stacked, 1, 2)
#         # for each in batch_embeddings:
#         hs = []
#         for block in self.cnn_blocks:
#             h = self[block](F.reshape(stacked, (batch_size, self.embed_units, -1)))
#             h = F.max_pooling_nd(h, max_word_length)
#             h = F.reshape(h, (batch_size, -1))
#             hs.append(h)
#
#         act = F.tanh(F.concat(hs, axis=1))
#         for highway in self.highways:
#             if self.inp_dropout > 0.:
#                 act = F.dropout(act, ratio=self.inp_dropout)
#             act = self[highway](act)
#         # Don't apply dropout to last layer
#         self.embedding = act
#
#         for i, word in enumerate(word_list):
#             self.cache[tuple(word)] = i
#
#     def word_to_index(self, word):
#         return self.cache[tuple(word)]
#
#     def clear_cache(self):
#         self.cache = dict()
