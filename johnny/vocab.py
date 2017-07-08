import six
import pickle
from collections import Counter, namedtuple
from johnny.dep import ROOT_REPR
# RESERVED = dict(ROOT=0, START=1, END=2)


class Vocab(object):
    """The tokens we know. Class defines a way to create the vocabulary
    and assign each known token to an index. All other tokens are replaced
    with the token UNK, which of course is UNK following the definition
    of Dr. UNK UNK from UNK.
    UNK is assigned the token 0 - because we like being arbitrary.
    The rest of the known tokens are sorted by frequency and assigned indices
    in such a manner.
    
    We keep the number of counts in order to be able to update our
    vocabulary later on. However, we throw away counts below or
    equal to threshold counts - because zipf's law and we don't
    have stocks in any companies producing ram chips.
    """

    special = dict(UNK=0, START=1, END=2)
    reserved = namedtuple('Reserved', special.keys())(**special)

    def __init__(self, size=None, out_size=None, counts=None, threshold=0):
        """
            size: int - the number of tokens we can represent.
            We always represent UNK, START and END but we don't count
            them in len. Use out_size attribute for that.

            counts: a dictionary of token, counts to initialise the vocab
            with.

            threshold: int - we throw away tokens with up to and including
            this many counts.
        """
        super(Vocab, self).__init__()
        if size is None:
            assert(out_size is not None)
            self.size = out_size - len(self.reserved)
            self.out_size = out_size
        elif out_size is None:
            assert(size is not None)
            self.out_size = size + len(self.reserved)
            self.size = size
        else:
            raise ValueError("Can't set both size and out_size")
        self.counts = counts or dict()
        self.threshold = threshold
        self.index = None
        self._threshold_counts()
        self._build_index()

    def __repr__(self):
        return ('Vocab object\ncapacity: %d\nactual size: %d\nthreshold: %d'
                % (self.size, len(self), self.threshold))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def _build_index(self):
        # we sort because in python 3 most_common is not guaranteed
        # to return the same order for elements with same count
        # when the code runs again. #fun_debugging
        candidates = sorted(self.counts.most_common(),
                            key=lambda x: (x[1], x[0]), reverse=True)
        limit = self.size
        offset = len(self.reserved)
        # we leave the 0 index to represent the UNK
        keep = candidates[:limit]
        if keep:
            keys, _ = zip(*keep)
            self.index = dict(zip(keys, range(offset, len(keys)+offset)))
        else:
            self.index = dict()

    def _threshold_counts(self):
        remove = []
        for key, c in six.iteritems(self.counts):
            if c <= self.threshold:
                remove.append(key)
        for key in remove:
            self.counts.pop(key)

    def encode(self, tokens, with_start=False, with_end=False):
        """tokens: iterable of tokens to get indices for.
        returns list of indices.
        """
        # We may insert START and END tokens before and after the
        # encoded sequence according to passed params
        return (((self.reserved.START,) if with_start else ()) +
                tuple(self.index.get(token, self.reserved.UNK)
                      for token in tokens) +
                ((self.reserved.END,) if with_end else ()))

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cl, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_token_list(cl, tokens, size=None, out_size=None, threshold=0):
        c = Counter(tokens)
        return cl(counts=c, size=size, out_size=out_size, threshold=threshold)


class UPOSVocab(object):
    """ Universal dependencies part of speech tag vocabulary.
    Alphabetical listing

    ADJ: adjective
    ADP: adposition
    ADV: adverb
    AUX: auxiliary
    CCONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other
    """
    TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
            'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']

    def __init__(self):
        super(UPOSVocab, self).__init__()
        self.tags = self.TAGS + [ROOT_REPR]
        self.index = dict((key, index) for index, key in enumerate(self.tags))

    def __repr__(self):
        return ('UPOSVocab object\nnum tags: %d\n' % (len(self), self.use_unk))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def encode(self, tags):
        """tags : iterable of tags """
        return tuple(self.index[tag] for tag in tags)


class UDepVocab(object):
    """ Universal dependency relations label vocabulary.
    Alphabetical listing

    acl: clausal modifier of noun (adjectival clause)
    advcl: adverbial clause modifier
    advmod: adverbial modifier
    amod: adjectival modifier
    appos: appositional modifier
    aux: auxiliary
    case: case marking
    cc: coordinating conjunction
    ccomp: clausal complement
    clf: classifier
    compound: compound
    conj: conjunct
    cop: copula
    csubj: clausal subject
    dep: unspecified dependency
    det: determiner
    discourse: discourse element
    dislocated: dislocated elements
    expl: expletive
    fixed: fixed multiword expression
    flat: flat multiword expression
    goeswith: goes with
    iobj: indirect object
    list: list
    mark: marker
    nmod: nominal modifier
    nsubj: nominal subject
    nummod: numeric modifier
    obj: object
    obl: oblique nominal
    orphan: orphan
    parataxis: parataxis
    punct: punctuation
    reparandum: overridden disfluency
    root: root
    vocative: vocative
    xcomp: open clausal complement
    """
    TAGS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
            'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj',
            'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat',
            'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod',
            'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root',
            'vocative', 'xcomp']

    def __init__(self):
        super(UDepVocab, self).__init__()
        self.tags = self.TAGS
        self.index = dict((key, index) for index, key in enumerate(self.tags))

    def __repr__(self):
        return ('UDepVocab object\nnum tags: %d' % (len(self), self.use_unk))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def encode(self, tags):
        """tags : iterable of tags """
        return tuple(self.index[tag] for tag in tags)
