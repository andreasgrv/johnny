import os
import six
import glob
import codecs
import json
import re
import numpy as np
import heapq
from itertools import chain
from collections import OrderedDict, defaultdict


def py2repr(f):
    def func(*args, **kwargs):
        x = f(*args, **kwargs)
        if six.PY2:
            return x.encode('utf-8')
        else:
            return x
    return func
    

class Dataset(object):

    def __init__(self, sents, lang=None, name=None):
        self.sents = sents
        self.lang = lang
        self.name = name

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

    def save(self, path):
        with codecs.open(path, 'w', encoding='utf-8') as inp:
            for sent in self:
                s = '%s\n\n' % '\n'.join(str(t) for t in sent)
                inp.write(s)

    @property
    def words(self):
        return tuple(s.words for s in self.sents)

    @property
    def heads(self):
        return tuple(s.heads for s in self.sents)

    @property
    def lemmas(self):
        return tuple(s.lemmas for s in self.sents)

    @property
    def feats(self):
        return tuple(s.feats for s in self.tokens)

    @property
    def arctags(self):
        return tuple(s.arctags for s in self.sents)

    @property
    def upostags(self):
        return tuple(s.upostags for s in self.sents)

    @property
    def xpostags(self):
        return tuple(s.xpostags for s in self.sents)

    @property
    def sent_lengths(self):
        return tuple(len(sent) for sent in self.sents)

    @property
    def arc_lengths(self):
        return tuple(chain(*[sent.arc_lengths for sent in self.sents]))

    def unset_heads(self):
        for s in self.sents:
            s.unset_heads()

    def unset_labels(self):
        for s in self.sents:
            s.unset_labels()

    def compute_token_ratios(self):
        # TODO: make this more efficient
        all_words = list(chain.from_iterable(s.words for s in self.sents))
        self.num_words = len(all_words)
        all_lemmas = list(chain.from_iterable(s.lemmas for s in self.sents))
        distinct_words = set(all_words)
        distinct_lemmas = set(all_lemmas)
        self.num_types = len(distinct_words)
        self.num_lemmas = len(distinct_lemmas)
        self.type_to_token_ratio = float(self.num_types)/self.num_words
        self.lemma_to_token_ratio = float(self.num_lemmas)/self.num_words

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
        self.compute_token_ratios()
        stats['num_words'] = self.num_words
        stats['num_types'] = self.num_types
        stats['num_lemmas'] = self.num_lemmas
        stats['type_to_token_ratio'] = self.type_to_token_ratio
        stats['lemma_to_token_ratio'] = self.lemma_to_token_ratio
        num_projective = sum(s.is_projective() for s in self.sents)
        stats['percentage_projective'] = float(num_projective)/len(self.sents)
        return stats


class Sentence(object):

    def __init__(self, tokens=None):
        # we are using this for dependency parsing
        # we don't care about multiword tokens or
        # repetition of words that won't be reflected in the sentence
        self.tokens = tuple(token for token in tokens
                            if token.head != -1) or tuple()

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

    def set_heads(self, heads):
        for t, h in zip(self.tokens, heads):
            t.head = h

    def set_labels(self, labels):
        for t, l in zip(self.tokens, labels):
            t.deprel = l

    def unset_heads(self):
        for t in self.tokens:
            t.head = None

    def unset_labels(self):
        for t in self.tokens:
            t.deprel = None

    def is_projective(self):
        """Check if this tree is projective """
        LEVEL_MARKER = -42 # why not?
        # parent to child map (don't need to use ids,
        # could use enumerate, but just in case there
        # is something particular with the ids..)
        p2c_map = defaultdict(list)
        for p, c in zip(self.heads, range(1, len(self.heads)+1)):
            p2c_map[p].append(c)
        visited = set()
        # I am (g)Rooooot!
        children = p2c_map[0]
        yield_stack = [[]]
        while children:
            child = children.pop()
            if child in visited:
                raise Exception('Cycle detected')
            # if we hit LEVEL_MARKER we are done processing a subtree
            # we therefore make sure the yield of the node has no gaps
            # if it does, the sentence is not projective
            if child == LEVEL_MARKER:
                node_yield = yield_stack.pop()
                node_yield = [heapq.heappop(node_yield) 
                              for i in range(len(node_yield))]
                prev = node_yield[0]
                for nxt in node_yield[1:]:
                    if nxt != (prev + 1):
                        return False
                    prev = nxt
                # if we made it to here, we merge the sorted lower subtree yield
                # with the nodes we had explored before visiting the subtree
                yield_stack[-1] = [heapq.heappop(yield_stack[-1])
                                   for i in range(len(yield_stack[-1]))]
                yield_stack[-1] = list(heapq.merge(yield_stack[-1], node_yield))
                # print(yield_stack)
                continue
            more_children = p2c_map[child]
            if more_children:
                # add a marker to know when we are done with the subtree
                children.append(LEVEL_MARKER)
                yield_stack.append([])
                children.extend(more_children)
            visited.add(child)
            heapq.heappush(yield_stack[-1], child)
            # print('heapq-push', child, yield_stack)
        return True

    @property
    def ids(self):
        return tuple(t.id for t in self.tokens)

    @property
    def words(self):
        return tuple(t.form for t in self.tokens)

    @property
    def heads(self):
        return tuple(t.head for t in self.tokens)

    @property
    def lemmas(self):
        return tuple(t.lemma for t in self.tokens)

    @property
    def feats(self):
        return tuple(t.feats for t in self.tokens)

    @property
    def arctags(self):
        return tuple(t.deprel.split(':')[0] for t in self.tokens)

    @property
    def upostags(self):
        return tuple(t.upostag for t in self.tokens)

    @property
    def xpostags(self):
        return tuple(t.xpostag for t in self.tokens)

    @property
    def arc_lengths(self):
        """Compute how long the arcs are in words"""
        return tuple(abs(head - index) if head != 0 else 1 for index, head in enumerate(self.heads, 1))


@six.python_2_unicode_compatible
class Token(object):

    # this is what each tab delimited attribute is expected to be
    # in the conllu data - exact order
    CONLLU_ATTRS = ['id', 'form', 'lemma', 'upostag', 'xpostag',
                    'feats', 'head', 'deprel', 'deps', 'misc']
    MORPH_SEP = '|'
    MORPH_ASSIGN = '='
    EMPTY = '_'
    # artificial morph join when feats not in key value pair form
    MORPH_FIX = ':'

    def __init__(self, *args):
        for i, prop in enumerate(args):
            label = Token.CONLLU_ATTRS[i]
            if label == 'head':
                # some words have _ as head when they are a multitoken representation
                # in that case replace with -1
                setattr(self, Token.CONLLU_ATTRS[i], int(prop)
                        if prop != self.EMPTY else -1)
            elif label == 'feats':
                if prop == self.EMPTY:
                    morph_dict = OrderedDict()
                else:
                    # some conll-x languages have a|b|c feats
                    # in that case we use the pos tag position of feat
                    if self.MORPH_ASSIGN not in prop:
                        morph_dict = OrderedDict(('%s%s%s' % (args[3],
                                                       self.MORPH_FIX,
                                                       i), m) 
                            for i, m in enumerate(prop.split(self.MORPH_SEP)))
                    else:
                        morph_dict = OrderedDict(m.split(self.MORPH_ASSIGN) 
                                for m in prop.split(self.MORPH_SEP))
                setattr(self, Token.CONLLU_ATTRS[i], morph_dict)
            else:
                setattr(self, Token.CONLLU_ATTRS[i], prop)

    @py2repr
    def __repr__(self):
        return '\t'.join(six.text_type(getattr(self, attr))
                         for attr in Token.CONLLU_ATTRS)

    def __str__(self):
        return '\t'.join(six.text_type(self.serialize(attr))
                         for attr in Token.CONLLU_ATTRS)

    def serialize(self, attr):
        value = getattr(self, attr)
        if attr == 'feats':
            if value:
                value = self.MORPH_SEP.join(self.MORPH_ASSIGN.join((key, val))
                                            if self.MORPH_FIX not in key else val
                                            for key, val in value.items())
            else:
                value = self.EMPTY
        return value

    
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

    CONLL2017 = 'CONLL2017'
    CONLL2006 = 'CONLL2006'
    AVAILABLE_LOADERS = [CONLL2017,
                         CONLL2006]

    def __init__(self, name, **kwargs):
        if name.startswith(self.CONLL2017):
            self.loader = CONLL2017Loader(name, **kwargs)
        elif name.startswith(self.CONLL2006):
            self.loader = CONLL2006Loader(name, **kwargs)
        else:
            raise ValueError('Unknown loader, name does not start with '
                    'one of %s' % self.AVAILABLE_LOADERS)

    def load_train_dev(self, lang, verbose=False):
        return self.loader.load_train_dev(lang, verbose=verbose)

    @staticmethod
    def get_env_var(name):
        return '%s_FOLDER' % name

    @staticmethod
    def load_conllu_sents(path):
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
                    cols = line.split('\t')
                    assert(len(cols) == 10)
                    tokens.append(Token(*cols))
        return sents

    @staticmethod
    def load_conllu(path):
        return Dataset(UDepLoader.load_conllu_sents(path))


class CONLL2006Loader(object):

    def __init__(self, name, datafolder=None, train_percentage=0.95):
        super(CONLL2006Loader, self).__init__()
        try:
            env_var = UDepLoader.get_env_var(name)
            self.datafolder = datafolder or os.environ[env_var]
            assert(os.path.isdir(self.datafolder))
        except KeyError:
            raise ValueError('You need to specify the path to the universal dependency '
                'root folder either using the datafolder argument or by '
                'setting the %s environment variable.' % env_var)
        self.train_percentage = train_percentage
        self.name = name
        file_path = os.path.join(self.datafolder, '*', '*', '*', '*', '*', '*.conll')
        file_paths = [f for f in glob.glob(file_path)]
        self.train_map = dict((os.path.basename(f).split('_', 1)[0], f)
                              for f in file_paths
                              if 'train' in f)
        # can't use 'test' since some files don't have test in the name!
        self.test_map = dict((os.path.basename(f).split('_', 1)[0], f)
                              for f in file_paths
                              if '_gs' in f)
        assert(len(self.train_map) == len(self.test_map))

    def __repr__(self):
        return ('<CONLL2006Loader object from folder %s with %d languages>'
                % (self.datafolder, len(self.langs)))

    def load_train_dev(self, lang, verbose=False):
        # we convert to lowercase to make matching easier
        p = self.train_map.get(lang.lower(), None)
        if p:
            sents = UDepLoader.load_conllu_sents(p) 
            # keep old state - we want the shuffling not to 
            # change whenever we change the seed
            rand_state = np.random.get_state()
            np.random.seed(62)
            np.random.shuffle(sents)
            # restore the state
            np.random.set_state(rand_state)
            num_sents = len(sents)
            split_index = int(num_sents * self.train_percentage)
            train = Dataset(sents[:split_index], lang=lang, name=self.name)
            dev = Dataset(sents[split_index:], lang=lang, name=self.name)
            if verbose:
                print('Loaded %d sentences from %s' % (len(train), p))
                print('Loaded %d sentences from %s' % (len(dev), p))
            return train, dev
        else:
            raise ValueError("Couldn't find a training file for %s"
                             % (lang))

    @property
    def langs(self):
        return list(six.viewkeys(self.train_map))


class CONLL2017Loader(object):
    LANG_FOLDER_REGEX = 'UD_(?P<lang>[A-Za-z\-\_]+)'
    PREFIX = 'UD_'
    TRAIN_SUFFIX = 'ud-train.conllu'
    DEV_SUFFIX = 'ud-dev.conllu'

    def __init__(self, name, datafolder=None):
        super(CONLL2017Loader, self).__init__()
        try:
            env_var = UDepLoader.get_env_var(name)
            self.datafolder = datafolder or os.environ[env_var]
        except KeyError:
            raise ValueError('You need to specify the path to the universal dependency '
                'root folder either using the datafolder argument or by '
                'setting the %s environment variable.' % env_var)
        self.lang_folders = dict()
        self.name = name
        found = False
        for lang_folder in os.listdir(self.datafolder):
            match = re.match(self.LANG_FOLDER_REGEX, lang_folder)
            if match:
                lang = match.groupdict()['lang']
                self.lang_folders[lang] = lang_folder
                found = True
        if not found:
            raise ValueError('No UD language folders '
                             'found in dir %s' % self.datafolder)

    def __repr__(self):
        return ('<CONLL2017Loader object from folder %s with %d languages>'
                % (self.datafolder, len(self.langs)))

    def load_train_dev(self, lang, verbose=False):
        p = os.path.join(self.datafolder, self.lang_folders[lang])
        train_filename = [fn for fn in os.listdir(p) 
                        if fn.endswith(self.TRAIN_SUFFIX)]
        if train_filename:
            train_filename = train_filename[0]
            train_path = os.path.join(p, train_filename)
            train = Dataset(UDepLoader.load_conllu_sents(train_path),
                            lang=lang, name=self.name)
            if verbose:
                print('Loaded %d sentences from %s' % (len(train), train_path))
        else:
            raise ValueError("Couldn't find a %s file for %s"
                             % (lang, self.TRAIN_SUFFIX))
        dev_filename = [fn for fn in os.listdir(p) 
                        if fn.endswith(self.DEV_SUFFIX)]
        if dev_filename:
            dev_filename = dev_filename[0]
            dev_path = os.path.join(p, dev_filename)
            dev = Dataset(UDepLoader.load_conllu_sents(dev_path),
                          lang=lang, name=self.name)
            if verbose:
                print('Loaded %d sentences from %s' % (len(dev), dev_path))
        else:
            raise ValueError("Couldn't find a %s file for %s"
                             % (lang, self.DEV_SUFFIX))
        return train, dev

    @property
    def langs(self):
        return list(six.viewkeys(self.lang_folders))
