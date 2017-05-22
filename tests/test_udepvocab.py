import pytest
from johnny.dep import UDepVocab

def test_basic():
    p = UDepVocab()
    e = p.encode(['acl', 'advcl'])
    assert(e == [p.TAGS.index('acl'), p.TAGS.index('advcl')])
    with pytest.raises(KeyError):
        e = p.encode(['acl', 'WTF'])
