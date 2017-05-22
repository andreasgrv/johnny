import codecs
import json
import six
from collections import Counter

# TODO : compare what we get from this loader with what we get from CONLL script

class Sentence(object):

    def __init__(self, tokens=None):
        # we are using this for dependency parsing
        # we don't care about multiword tokens or
        # repetition of words that won't be reflected in the sentence
        self.tokens = [token for token in tokens if token.head != -1] or []

    def __getitem__(self, index):
        return self.tokens[index]

    def __repr__(self):
        return ' '.join(token.form for token in self.tokens).encode('utf-8')

    def __len__(self):
        return len(self.tokens)

    def displacify(self, universal_pos=True):
        arcs = [token.displacy_arc() for token in self.tokens
                if token.deprel != 'root' and token.displacy_arc()]
        words = [token.displacy_word(universal_pos=universal_pos)
                 for token in self.tokens if token.displacy_word()]
        return json.dumps(dict(arcs=arcs, words=words))

    @property
    def heads(self):
        return [t.head for t in self.tokens]

    @property
    def utags(self):
        return [t.upostag for t in self.tokens]

    @property
    def xtags(self):
        return [t.xpostag for t in self.tokens]

class Token(object):

    # this is what each tab delimited attribute is expected to be
    # in the conllu data - exact order
    CONLLU_ATTRS = ['id', 'form', 'lemma', 'upostag', 'xpostag',
                    'feats', 'head', 'deprel', 'deps', 'misc']
    def __init__(self, *args):
        for i, prop in enumerate(args):
            label = Token.CONLLU_ATTRS[i]
            if label in ['head']:
                # some words have _ as head when they are a multitoken representation
                # in that case replace with -1
                setattr(self, Token.CONLLU_ATTRS[i], int(prop) if prop != '_' else -1)
            else:
                setattr(self, Token.CONLLU_ATTRS[i], prop)

    def __repr__(self):
        return '\t'.join(getattr(self, attr) for attr in Token.CONLLU_ATTRS).encode('utf-8')
    
    def displacy_word(self, universal_pos=True):
        """ return a dictionary that matches displacy format """
        tag = self.upostag if universal_pos else self.xpostag
        if '-' not in self.id:
            return dict(tag=tag, text=self.form)
        else:
            return dict()

    def displacy_arc(self):
        """ return a dictionary that matches displacy format """
        # sometimes id, head can be two numbers separated by - : 4-5
        try:
            start_i, end_i = (int(self.id) - 1, int(self.head) - 1)
            start, end, direction =  (start_i, end_i, 'left') if start_i <= end_i else (end_i, start_i, 'right')
            return dict(start=start, end=end, dir=direction, label=self.deprel)
        except Exception:
            return dict()

def load_conllu(filename):
    """ Read in conll file and return a list of sentences """
    CONLLU_COMMENT = '#'
    sents = []
    with codecs.open(filename, 'r', encoding='utf-8') as inp:
        tokens = []
        for line in inp:
            line = line.rstrip()
            # we ignore documents for the time being
            if tokens and not line:
                sents.append(Sentence(tokens))
                tokens = []
            if line and not line.startswith(CONLLU_COMMENT):
                tokens.append(Token(*line.split('\t')))
    return sents


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

    UNK = 0

    def __init__(self, size, counts=None, threshold=0):
        """
            size: int - the number of tokens we can represent - also the
            size of the index. We also represent UNK but that is not accounted
            for in size (we don't store UNK in the index).

            counts: a dictionary of token, counts to initialise the vocab
            with.

            threshold: int - we throw away tokens with up to and including
            this many counts.
        """
        super(Vocab, self).__init__()
        self.counts = counts or dict()
        self.size = size
        self.threshold = threshold
        self.index = None
        self._threshold_counts()
        self._build_index()

    def __repr__(self):
        return ('Vocab object\ncapacity: %d\nactual size: %d\nthreshold: %d'
                % (self.size, len(self), self.threshold))

    def __len__(self):
        return len(self.index) # we don't store UNK so that is not in the count

    def __getitem__(self, key):
        return self.index[key]

    def _build_index(self):
        # we leave the 0 index to represent the UNK
        keep = self.counts.most_common(self.size - 1)
        if keep:
            keys, _ = zip(*keep)
            self.index = dict(zip(keys, range(1, len(keys)+1)))
        else:
            self.index = dict()

    def _threshold_counts(self):
        remove = []
        for key, c in six.iteritems(self.counts):
            if c <= self.threshold:
                remove.append(key)
        for key in remove:
            self.counts.pop(key)

    def encode(self, tokens):
        """tokens: iterable of tokens to get indices for.
        returns list of indices.
        """
        return [self.index.get(token, self.UNK) for token in tokens]

    @classmethod
    def from_token_list(cl, tokens, size, threshold=0):
        c = Counter(tokens)
        return cl(size, counts=c, threshold=threshold)


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
        self.tags = self.TAGS
        self.index = dict([(key, index) for index, key in enumerate(self.tags)])

    def __repr__(self):
        return ('UPOSVocab object\nnum tags: %d\nuse_unk: %s'
                % (len(self), self.use_unk))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def encode(self, tags):
        """tags : iterable of tags """
        return [self.index[tag] for tag in tags]
