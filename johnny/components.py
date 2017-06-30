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
        self.dropout = dropout
        with self.init_scope():
            for index, (in_size, out_size) in enumerate(zip(in_sizes, out_sizes)):
                embed_layer = L.EmbedID(in_size, out_size, ignore_label=-1)
                self.set_embed(index, embed_layer)
        self.out_size = sum(out_sizes)

    def get_embed(self, index):
        return self['embed_%d' % index]

    def set_embed(self, index, embed):
        setattr(self, 'embed_%d' % index, embed)

    def __call__(self, *seqs):
        act = F.concat((self.get_embed(i)(s) for i, s in enumerate(seqs)), axis=1)
        if self.dropout > 0:
            act = F.dropout(act, ratio=self.dropout)
        return act
