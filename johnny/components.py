from itertools import chain
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.links.connection.n_step_lstm as chainer_nstep
from johnny.extern import NStepLSTMBase
from johnny.vocab import augment_seq, augment_seq_nested, augment_word, reserved

CHAINER_IGNORE_LABEL = -1


class Embedder(chainer.Chain):
    """A general embedder that concatenates the embeddings of an arbitrary
    number of input sequences."""

    def __init__(self, in_sizes, out_sizes, dropout=0.2):
        """
        :in_sizes: list of ints specifying the input vocabulary size for each
        sequence.
        :out_sizes: list of ints specifying embedding size for each sequence.
        :dropout: float between 0 and 1, how much dropout to apply to the input

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


class SentenceEncoder(chainer.Chain):
    """Encodes a sentence word by word using a recurrent neural network."""


    CHAINER_IGNORE_LABEL = -1

    def __init__(self, embedder, use_bilstm=True, num_layers=1,
                 num_units=100, dropout=0.2):

        super(SentenceEncoder, self).__init__()
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
        max_seq_len = len(seqs[0])
        if create_var:
            batch = [chainer.Variable(self.xp.array([sent[i]
                                                    for sent in seqs
                                                    if i < len(sent)],
                                      dtype=self.xp.int32))
                     for i in range(max_seq_len)]
        else:
            batch = [self.xp.array([sent[i]
                                    for sent in seqs
                                    if i < len(sent)],
                                   dtype=self.xp.int32)
                     for i in range(max_seq_len)]
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
        # we add 1 because we are augmenting the sentence with a ROOT symbol 
        self.max_seq_len = len(sents[0]) + 1
        self.batch_size = len(sents)

        # before we modify input we check if our embedder handles subword
        # information. if so we want to pass the list of individual words
        # to the embedder for efficiency purposes (we can encode each word
        # and employ a lookup table on the actual forward pass for each
        # batch). We assume the subword tokens are passed in a 3D array
        # as in_seqs[0] -> sentences x words in sentence x tokens in words
        if getattr(self.embedder, 'is_subword', False):
            # pad each word with START_WORD END_WORD
            sents = tuple(tuple(map(augment_word, s)) for s in sents)
            # unique list of words sorted from longest to shortest
            word_set = set(chain.from_iterable(sents))
            # also include the sentence level markers
            # as single words [token]
            word_set.update([(reserved.START_SENTENCE,),
                             (reserved.END_SENTENCE,),
                             (reserved.ROOT,)])
            word_encoder = self.embedder.word_encoder
            word_encoder.encode_words(word_set)
            # replace 3D input with 2D - words replaced with hash
            fwd_first = tuple(tuple(map(word_encoder.word_to_index,
                                        augment_seq_nested(s)))
                              for s in sents)
            fwd_rest = [self.transpose_batch(tuple(map(augment_seq, seq)))
                        for seq in in_seqs[1:]]
            fwd = [self.transpose_batch(fwd_first)]
            fwd.extend(fwd_rest)
        else:
            # we augment sequence here - augmenting with START, ROOT at the
            # beginning and END at the you know where
            # turn batch_size x seq_len -> seq_len x batch_size
            # NOTE: seq_len is variable - we aren't padding
            fwd = [self.transpose_batch(tuple(map(augment_seq, seq)))
                   for seq in in_seqs]

        # collapse all ids into a vector
        embeddings = self.embedder(*(F.concat(f, axis=0) for f in fwd))

        col_lengths = [len(col) for col in fwd[0]]

        # use np because cumsum crashes gpu - I know, right?
        batch_split = np.cumsum(col_lengths[:-1])
        # split back to batch size
        batch_embeddings = F.split_axis(embeddings, batch_split, axis=0)

        _, _, states = self.rnn(None, None, batch_embeddings)

        # we don't use the START and END encoded states in attention
        # so we get rid of them from states and col_lengths
        # also creates col_lengths
        # states = self.deaugment(states)
        keep = list()
        self.col_lengths = []
        # END tokens are spread out across the matrix since
        # we have variable length inputs (sorted)
        # eg:  S R 1 2 3 4 E     We want to get rid of S and E without
        #      S R 5 6 E         transposing and stuff (we want to keep
        #      S R 7 E           the root embedding R)
        #
        # to:  1 2 3 4
        #      5 6
        #      7
        for i in range(1, len(states) - 1):
            # if the next column is shorter, it means that in the original
            # data this column was shorter. If not clear imagine a layer of
            # blocks falling when playing tetris.
            diff = len(states[i]) - len(states[i+1])
            if diff > 0:
                clean = states[i][:-diff]
            else:
                clean = states[i]
            col_len = len(clean)
            self.col_lengths.append(col_len)
            keep.append(F.pad(clean,
                              ((0, self.batch_size - col_len), (0,0)),
                              'constant',
                              constant_values=0.))
        states = F.vstack(keep)
        # discard first and last column. The first column always contains
        # START. The last column contains only END but END tokens are
        # spread throughout.

        # states = F.vstack((F.pad(s,
        #                          ((0, self.batch_size - len(s)), (0,0)),
        #                          'constant',
        #                          constant_values=0.)
        #                    for s in states))

        mask_shape = (self.batch_size, self.max_seq_len)
        self.mask = self.xp.ones(mask_shape, dtype=self.xp.bool_)
        for i, l in enumerate(self.col_lengths):
            self.mask[l:, i] = False

        if getattr(self.embedder, 'is_subword', False):
            # Remember to clear cache
            self.embedder.word_encoder.clear_cache()

        # returns 2d (batch_size x max_sentence_length) x num_hidden
        return states


class LSTMWordEncoder(chainer.Chain):

    def __init__(self, vocab_size, num_units, num_layers,
                 inp_dropout=0.2, rec_dropout=0.2, use_bilstm=True):

        super(LSTMWordEncoder, self).__init__()
        with self.init_scope():
            self.embed_layer = L.EmbedID(vocab_size, num_units,
                                         ignore_label=CHAINER_IGNORE_LABEL)
            self.rnn = chainer_nstep.NStepLSTMBase(num_layers,
                                                   num_units,
                                                   num_units,
                                                   rec_dropout,
                                                   None,
                                                   None,
                                                   use_bi_direction=use_bilstm)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.rec_dropout = rec_dropout
        self.inp_dropout = inp_dropout
        self.use_bilstm = use_bilstm
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

        if self.inp_dropout > 0.:
            embeddings = F.dropout(embeddings, ratio=self.inp_dropout)

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


class CNNWordEncoder(chainer.Chain):

    FILTER_MULTIPLIER = 25
    IGNORE_LABEL = -1

    def __init__(self, vocab_size, embed_units=15, num_highway_layers=1,
                 highway_dropout=0.0, ngrams=(1, 2, 3, 4, 5, 6), stride=1,
                 num_filters=None, batch_norm=False):

        super(CNNWordEncoder, self).__init__()
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
            self.min_width = max(ngrams)
            self.highways = ['highway_%d' % i for i in range(num_highway_layers)]
            # for n in ngrams:
            #     setattr(self, self.cnn_blocks[n])
            for i, name in enumerate(self.cnn_blocks):
                setattr(self, name, L.Convolution2D(1,
                        num_filters[i],
                        (ngrams[i], embed_units),
                        stride))
            if batch_norm:
                self.batch_norm = L.BatchNormalization(out_size)
            for name in self.highways:
                # init_bt -2 used in Kim paper
                setattr(self, name, L.Highway(out_size, init_bt=-2))

        self.vocab_size = vocab_size
        self.embed_units = embed_units
        self.num_highway_layers = num_highway_layers
        self.highway_dropout = highway_dropout
        self.use_batch_norm = batch_norm
        # highway doesn't change dimensionality
        self.out_size = out_size
        self.cache = dict()

    def __call__(self, batch):
        return self.embedding[batch.data]

    def encode_words(self, word_list):

        batch_size = len(word_list)
        width = np.max(tuple(map(len, word_list)))
        if width < self.min_width:
            width = self.min_width
        word_vars = F.concat([chainer.Variable(self.xp.array([w[i]
                                               if i < len(w)
                                               else reserved.PAD
                                               for i in range(width)],
                                               dtype=self.xp.int32))
                              for w in word_list], axis=0)

        embeddings = self.embed_layer(word_vars)

        stacked = F.reshape(embeddings, (batch_size, -1, self.embed_units))

        stacked = F.expand_dims(stacked, axis=1)

        # for each in batch_embeddings:
        acts = []
        for block in self.cnn_blocks:
            h = self[block](stacked)
            h = F.relu(h)
            # NOTE: batch_size is num_words in batch
            # h shape : batch_size x num_filters x sent_len x 1
            # =============================================================
            # h = F.max_pooling_2d(h, (width,
            #                          self.embed_units))
            # =============================================================
            # NOTE: below max is max pooling over "time".
            # we are practically only keeping the max over the sequence
            # so we don't need to use the max pooling 2d function
            # which is much slower (especially on cpu)
            h = F.max(h, 2)
            # collapse last dimension
            h = F.squeeze(h, 2)
            # h shape : batch_size x num_filters
            acts.append(h)
        act = F.concat(acts, axis=1)
        if self.use_batch_norm:
            act = self.batch_norm(act)

        # Don't apply dropout to last layer
        for highway in self.highways:
            if self.highway_dropout > 0.:
                act = F.dropout(act, ratio=self.highway_dropout)
            act = self[highway](act)

        self.embedding = act

        for i, word in enumerate(word_list):
            self.cache[tuple(word)] = i

    def word_to_index(self, word):
        return self.cache[tuple(word)]

    def clear_cache(self):
        self.cache = dict()


# class AltCNNWordEncoder(chainer.Chain):
#
#     FILTER_MULTIPLIER = 25
#     IGNORE_LABEL = -1
#
#     def __init__(self, vocab_size, embed_units=15, num_highway_layers=1,
#                  highway_dropout=0.0, ngrams=(1, 2, 3, 4, 5, 6), stride=1, num_filters=None):
#
#         super(AltCNNWordEncoder, self).__init__()
#         if num_filters is None:
#             # http://www.people.fas.harvard.edu/~yoonkim/data/char-nlm.pdf
#             # Table 2 small model uses constant size
#             num_filters = [n * self.FILTER_MULTIPLIER for n in ngrams]
#         assert(len(num_filters) == len(ngrams))
#         assert(num_highway_layers >= 0)
#         out_size = sum(num_filters)
#         with self.init_scope():
#             self.embed_layer = L.EmbedID(vocab_size, embed_units,
#                                          ignore_label=self.IGNORE_LABEL)
#             self.cnn_blocks = ['cnn_%d' % n for n in ngrams]
#             self.min_width = max(ngrams)
#             self.highways = ['highway_%d' % i for i in range(num_highway_layers)]
#             # for n in ngrams:
#             #     setattr(self, self.cnn_blocks[n])
#             for i, name in enumerate(self.cnn_blocks):
#                 setattr(self, name, L.Convolution2D(embed_units,
#                                        num_filters[i],
#                                        (ngrams[i], 1),
#                                        stride))
#             for name in self.highways:
#                 # init_bt -2 used in Kim paper
#                 setattr(self, name, L.Highway(out_size, init_bt=-2))
#         self.vocab_size = vocab_size
#         self.embed_units = embed_units
#         self.num_highway_layers = num_highway_layers
#         self.highway_dropout = highway_dropout
#         # highway doesn't change dimensionality
#         self.out_size = out_size
#         self.cache = dict()
#
#     def __call__(self, batch):
#         return self.embedding[batch.data]
#
#     def encode_words(self, word_list):
#
#         batch_size = len(word_list)
#         sorted_word_list = sorted(word_list, key=lambda x: len(x), reverse=True)
#
#         # split into parts - because there will be very few very long words
#         # might as well pack them together to avoid wasting computation on padding
#         # NOTE: batch size here is number of words in each batch of encoder
#         # so for 32 batch size this can be 1000
#         SPLIT_INDEX = int(0.1 * batch_size)
#         if SPLIT_INDEX > 0:
#             batch_split = np.array_split(sorted_word_list, [SPLIT_INDEX])
#         else:
#             batch_split = [sorted_word_list]
#
#         acts = None
#         for shard in batch_split:
#
#             shard_len = len(shard)
#             shard_longest_word = len(shard[0])
#             width = shard_longest_word
#             if width < self.min_width:
#                 width = self.min_width
#             word_vars = F.concat([chainer.Variable(self.xp.array([w[i]
#                                                    if i < len(w)
#                                                    else reserved.PAD
#                                                    for i in range(width)],
#                                                    dtype=self.xp.int32))
#                                           for w in shard], axis=0)
#
#             embeddings = self.embed_layer(word_vars)
#
#             stacked = F.reshape(embeddings, (shard_len, -1, self.embed_units))
#
#             stacked = F.expand_dims(stacked, axis=1)
#
#             # for each in batch_embeddings:
#             act = None
#             for block in self.cnn_blocks:
#                 stacked = F.reshape(stacked, (shard_len, self.embed_units, -1, 1))
#                 # print(stacked.shape)
#                 h = self[block](stacked)
#                 h = F.tanh(h)
#                 # print(h.shape)
#                 # NOTE: batch_size is num_words in batch
#                 # h shape : batch_size x num_filters x sent_len x 1
#                 # =============================================================
#                 # h = F.max_pooling_2d(h, (width,
#                 #                          self.embed_units))
#                 # =============================================================
#                 # NOTE: below max is max pooling over "time".
#                 # we are practically only keeping the max over the sequence
#                 # so we don't need to use the max pooling 2d function
#                 # which is much slower (especially on cpu)
#                 # print(h.shape)
#                 h = F.max(h, 2)
#                 # collapse last dimension
#                 h = F.squeeze(h, 2)
#                 # h shape : batch_size x num_filters
#                 # print(h.shape)
#                 if act is None:
#                     act = h
#                 else:
#                     # stack ngram filter activations along num_filters axis
#                     act = F.concat([act, h], axis=1)
#
#             # Don't apply dropout to last layer
#             for highway in self.highways:
#                 if self.highway_dropout > 0.:
#                     act = F.dropout(act, ratio=self.highway_dropout)
#                 act = self[highway](act)
#             if acts is None:
#                 acts = act
#             else:
#                 acts = F.vstack([acts, act])
#
#         self.embedding = acts
#
#         for i, word in enumerate(sorted_word_list):
#             self.cache[tuple(word)] = i
#
#     def word_to_index(self, word):
#         return self.cache[tuple(word)]
#
#     def clear_cache(self):
#         self.cache = dict()
