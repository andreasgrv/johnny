import pytest
from johnny.vocab import UPOSVocab

def test_basic():
    p = UPOSVocab()
    e = p.encode(['X', 'NOUN'])
    assert(e == (p.TAGS.index('X'), p.TAGS.index('NOUN')))
    with pytest.raises(KeyError):
        e = p.encode(['X', 'WTF'])
