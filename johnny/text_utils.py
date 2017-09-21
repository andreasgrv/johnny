""" Preprocessing and text encoding """
import re
import unicodedata
from itertools import chain


COLLAPSE_NUMS_RE = re.compile(r'\d*[\.,]?\d+')
COLLAPSE_TRIPLE_RE = re.compile(r'(.)\1{2,}')


def _collapse_nums(s, repl='__NUM__'):
    """Replace numbers with __NUM__"""
    return COLLAPSE_NUMS_RE.sub(repl, s)


def _collapse_triples(s):
    """Truncate repetitions of length 3 or more to 2 characters.
    
    example: baaaggggg -> baagg

    """
    return COLLAPSE_TRIPLE_RE.sub(r'\1\1', s)


def _expand_diacritics(s):
    """Expand diacritics to separate tokens

    example: ταΐζω -> ται¨´ζω

    """
    return unicodedata.normalize('NFD', s)


def _remove_diacritics(s):
    """Expand diacritics to separate tokens

    example: ταΐζω -> ται¨´ζω

    """
    return ''.join(unicodedata.normalize('NFD', c)[0] for c in s)


def preprocess(word,
               lowercase=False,
               collapse_nums=False,
               collapse_triples=False,
               remove_diacritics=False,
               expand_diacritics=False):
    if lowercase:
        word = word.lower()
    # replace numbers with __NUM__
    if collapse_nums:
        word = _collapse_nums(word)
    # collapse more than 3 repetitions of a character
    # to two repetitions TODO: maybe this is bad for some langs?
    if collapse_triples:
        word = _collapse_triples(word)
    if remove_diacritics:
        word = _remove_diacritics(word)
    if expand_diacritics:
        word = _expand_diacritics(word)
    return word


def to_ngrams(iterable, n=1, pad='_'):
    assert n > 0, 'value of n must be greater than 0'
    assert len(iterable) >= 1
    if n == 1:
        return tuple(e for e in iterable)
    # if n is more than the length of the iterable - pad it
    if n > len(iterable):
        return (tuple(chain(iterable,[pad] * (n - len(iterable)))),)
    else:
        return tuple(tuple(iterable[i:i+n])
                     for i in range(len(iterable)-n+1))


def process_text(sent, ngram=1, is_subword=False, preprocess_funcs=None):
    """Preprocess text and create ngrams from an iterable of tokens.

    sent: iterable of tokens

    returns: generator expression of ngrams of preprocessed sentence.
    """
    if preprocess_funcs is None:
        preprocess_funcs = dict()
    if is_subword:
        return tuple(to_ngrams(preprocess(w, **preprocess_funcs),
                               n=ngram)
                     for w in sent)
    else:
        return tuple(to_ngrams(tuple(preprocess(w, **preprocess_funcs) for w in sent),
                               n=ngram))


def encode_texts(sents, vocab, is_subword=False):
    """Encode a batch of sentences using a vocabulary.
    This converts the strings to ids looked up in the vocab
    dictionary."""
    if is_subword:
        return tuple(tuple(map(vocab.encode, s)) for s in sents)
    else:
        return tuple(map(vocab.encode, sents))
