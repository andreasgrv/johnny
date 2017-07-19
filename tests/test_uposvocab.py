import pytest
from johnny.vocab import UPOSVocab, RESERVED

def test_basic():
    p = UPOSVocab()
    e = p.encode(['X', 'NOUN'])
    num_reserved = len(RESERVED)
    assert(e == (p.TAGS.index('X') + num_reserved,
                 p.TAGS.index('NOUN') + num_reserved))
    with pytest.raises(KeyError):
        e = p.encode(['X', 'WTF'])
