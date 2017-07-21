# -*- coding: utf-8 -*-
import johnny.preprocess as pp

def test_collapse_nums():
    s = 'It cost 54.3$'
    repl = '__NUM__'
    assert(pp.collapse_nums(s, repl) == 'It cost %s$' % repl)
    s = '5000 things'
    assert(pp.collapse_nums(s, repl) == '%s things' % repl)
    s = '13th of December'
    assert(pp.collapse_nums(s, repl) == '%sth of December' % repl)
    s = '13 1556'
    assert(pp.collapse_nums(s, repl) == '%s %s' % (repl, repl))

def test_collapse_triples():
    s = 'peeeeerfect!!!!+++'
    assert(pp.collapse_triples(s) == 'peerfect!!++')

def test_expand_diacritics():
    s = u'ταΐζω'
    assert(len(pp.expand_diacritics(s)) == len(s) + 2)
    s = u'ᾧ'
    assert(len(pp.expand_diacritics(s)) == len(s) + 3)

def test_remove_diacritics():
    s = u'ταΐζω'
    assert(pp.remove_diacritics(s) == u'ταιζω')
    s = u'ᾧ'
    assert(pp.remove_diacritics(s) == u'ω')
