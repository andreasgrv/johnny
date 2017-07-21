# -*- coding: utf-8 -*-
import re
import unicodedata

COLLAPSE_NUMS_RE = re.compile(r'\d*[\.,]?\d+')
COLLAPSE_TRIPLE_RE = re.compile(r'(.)\1{2,}')

def collapse_nums(s, repl='__NUM__'):
    """Replace numbers with __NUM__"""
    return COLLAPSE_NUMS_RE.sub(repl, s)

def collapse_triples(s):
    """Truncate repetitions of length 3 or more to 2 characters.
    
    example: baaaggggg -> baagg

    """
    return COLLAPSE_TRIPLE_RE.sub(r'\1\1', s)

def expand_diacritics(s):
    """Expand diacritics to separate tokens

    example: ταΐζω -> ται¨´ζω

    """
    return unicodedata.normalize('NFD', s)

def remove_diacritics(s):
    """Expand diacritics to separate tokens

    example: ταΐζω -> ται¨´ζω

    """
    return ''.join(unicodedata.normalize('NFD', c)[0] for c in s)
