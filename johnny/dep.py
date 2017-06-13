import os
import six
import codecs
import json
import re
import numpy as np
from collections import Counter
from itertools import chain
from johnny import DATA_ENV_VAR

# TODO : compare what we get from this loader with what we get from CONLL script

ROOT_REPR = '__ROOT__'


def py2repr(f):
    def func(*args, **kwargs):
        x = f(*args, **kwargs)
        if six.PY2:
            return x.encode('utf-8')
        else:
            return x
    return func
    
# TODO: Create Dataset class - that allows stats viewing.
# the Dataset class should be a list of sentences enhanced with
# properties such as words, heads, pos etc. 
# the loader should return a Dataset object instead of
# a list of sentences

class Dataset(object):

    def __init__(self, sents, lang=None):
        self.sents = sents
        self.lang = lang

    def __getitem__(self, index):
        return self.sents[index]

    def __len__(self):
        return len(self.sents)

    @py2repr
    def __repr__(self):
        return 'Dataset of %s - %s sents' % (self.lang, len(self.sents))

    def __iter__(self):
        for s in self.sents:
            yield s

    @property
    def words(self):
        return [s.words for s in self.sents]

    @property
    def heads(self):
        return [s.heads for s in self.sents]

    @property
    def arctags(self):
        return [s.arctags for s in self.sents]

    @property
    def upostags(self):
        return [s.upostags for s in self.sents]

    @property
    def xpostags(self):
        return [s.xpostags for s in self.sents]

    @property
    def sent_lengths(self):
        return [len(sent) for sent in self.sents]

    @property
    def arc_lengths(self):
        return list(chain(*[sent.arc_lengths for sent in self.sents]))

    @property
    def len_stats(self):
        sent_lens = self.sent_lengths
        self.max_sent_len = max(sent_lens)
        self.min_sent_len = min(sent_lens)
        self.avg_sent_len = np.mean(sent_lens)
        self.std_sent_len = np.std(sent_lens)
        return {'max_sent_len': self.max_sent_len,
                'min_sent_len': self.min_sent_len,
                'avg_sent_len': self.avg_sent_len,
                'std_sent_len': self.std_sent_len}

    @property
    def arc_len_stats(self):
        arc_lengths = self.arc_lengths 
        self.max_arc_len = max(arc_lengths)
        self.min_arc_len = min(arc_lengths)
        self.avg_arc_len = np.mean(arc_lengths)
        self.std_arc_len = np.std(arc_lengths)
        return {'max_arc_len': self.max_arc_len,
                'min_arc_len': self.min_arc_len,
                'avg_arc_len': self.avg_arc_len,
                'std_arc_len': self.std_arc_len}

    @property
    def stats(self):
        stats = dict(**self.len_stats)
        stats.update(**self.arc_len_stats)
        stats['num_sents'] = len(self)
        return stats


class Sentence(object):


    def __init__(self, tokens=None):
        # we are using this for dependency parsing
        # we don't care about multiword tokens or
        # repetition of words that won't be reflected in the sentence
        self.tokens = [token for token in tokens if token.head != -1] or []

    def __getitem__(self, index):
        return self.tokens[index]

    def __iter__(self):
        for t in self.tokens:
            yield t

    @py2repr
    def __repr__(self):
        return ' '.join(token.form for token in self.tokens)

    def __len__(self):
        return len(self.tokens)

    def displacify(self, universal_pos=True):
        arcs = [token.displacy_arc() for token in self.tokens
                if token.deprel != 'root' and token.displacy_arc()]
        words = [token.displacy_word(universal_pos=universal_pos)
                 for token in self.tokens if token.displacy_word()]
        return json.dumps(dict(arcs=arcs, words=words))

    @property
    def words(self):
        words = [ROOT_REPR]
        words.extend([t.form for t in self.tokens])
        return words

    @property
    def heads(self):
        return [t.head for t in self.tokens]

    @property
    def arctags(self):
        return [t.deprel.split(':')[0] for t in self.tokens]

    @property
    def upostags(self):
        tags = [ROOT_REPR]
        tags.extend([t.upostag for t in self.tokens])
        return tags

    @property
    def xpostags(self):
        tags = [ROOT_REPR]
        tags.extend([t.xpostag for t in self.tokens])
        return tags

    @property
    def arc_lengths(self):
        """Compute how long the arcs are in words"""
        return [abs(head - index) if head != 0 else 1 for index, head in enumerate(self.heads, 1)]


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

    @py2repr
    def __repr__(self):
        return '\t'.join(six.text_type(getattr(self, attr)) for attr in Token.CONLLU_ATTRS)
    
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


class UDepLoader(object):
    """Loader for universal dependencies datasets"""
    LANG_FOLDER_REGEX = 'UD_(?P<lang>[A-Za-z\-\_]+)'
    PREFIX = 'UD_'
    TRAIN_SUFFIX = 'ud-train.conllu'
    DEV_SUFFIX = 'ud-dev.conllu'

    def __init__(self, datafolder=None):
        super(UDepLoader, self).__init__()
        try:
            self.datafolder = datafolder or os.environ[DATA_ENV_VAR]
        except KeyError:
            raise ValueError('You need to specify the path to the universal dependency '
                'root folder either using the datafolder argument or by '
                'setting the %s environment variable.' % self.DATA_ENV_VAR)
        self.lang_folders = dict()
        for lang_folder in os.listdir(self.datafolder):
            match = re.match(self.LANG_FOLDER_REGEX, lang_folder)
            lang = match.groupdict()['lang']
            self.lang_folders[lang] = lang_folder

    def __repr__(self):
        return ('<UDepLoader object from folder %s with %d languages>'
                % (self.datafolder, len(self.langs)))

    @staticmethod
    def load_conllu(path):
        """ Read in conll file and return a list of sentences """
        CONLLU_COMMENT = '#'
        sents = []
        with codecs.open(path, 'r', encoding='utf-8') as inp:
            tokens = []
            for line in inp:
                line = line.rstrip()
                # we ignore documents for the time being
                if tokens and not line:
                    sents.append(Sentence(tokens))
                    tokens = []
                if line and not line.startswith(CONLLU_COMMENT):
                    tokens.append(Token(*line.split('\t')))
        return Dataset(sents)

    def load_train(self, lang, verbose=False):
        p = os.path.join(self.datafolder, self.lang_folders[lang])
        train_filename = [fn for fn in os.listdir(p) 
                        if fn.endswith(self.TRAIN_SUFFIX)]
        if train_filename:
            train_filename = train_filename[0]
            train_path = os.path.join(p, train_filename)
            sents = self.load_conllu(train_path) 
            if verbose:
                print('Loaded %d sentences from %s' % (len(sents), train_path))
            return Dataset(sents)
        else:
            raise ValueError("Couldn't find a %s file for %s"
                             % (lang, self.TRAIN_SUFFIX))

    def load_dev(self, lang, verbose=False):
        p = os.path.join(self.datafolder, self.lang_folders[lang])
        dev_filename = [fn for fn in os.listdir(p) 
                        if fn.endswith(self.DEV_SUFFIX)]
        if dev_filename:
            dev_filename = dev_filename[0]
            dev_path = os.path.join(p, dev_filename)
            sents = self.load_conllu(dev_path) 
            if verbose:
                print('Loaded %d sentences from %s' % (len(sents), dev_path))
            return sents
        else:
            raise ValueError("Couldn't find a %s file for %s"
                             % (lang, self.DEV_SUFFIX))

    @property
    def langs(self):
        return list(six.viewkeys(self.lang_folders))


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
        # we sort because in python 3 most_common is not guaranteed
        # to return the same order for elements with same count
        # when the code runs again. #fun_debugging
        candidates = sorted(self.counts.most_common(),
                            key=lambda x: (x[1], x[0]), reverse=True)
        limit = self.size - 1 if self.size > 0 else 0
        # we leave the 0 index to represent the UNK
        keep = candidates[:limit]
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
        self.tags = self.TAGS + [ROOT_REPR]
        self.index = dict([(key, index) for index, key in enumerate(self.tags)])

    def __repr__(self):
        return ('UPOSVocab object\nnum tags: %d\n' % (len(self), self.use_unk))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def encode(self, tags):
        """tags : iterable of tags """
        return [self.index[tag] for tag in tags]


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
        self.index = dict([(key, index) for index, key in enumerate(self.tags)])

    def __repr__(self):
        return ('UDepVocab object\nnum tags: %d' % (len(self), self.use_unk))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def encode(self, tags):
        """tags : iterable of tags """
        return [self.index[tag] for tag in tags]
