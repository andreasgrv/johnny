from johnny.dep import Sentence
from collections import namedtuple


def test_arclen():
    t = namedtuple('TokenStub', ('head'))
    s = Sentence([t(3), t(1), t(0), t(2)])
    assert(s.arc_lengths == [2, 1, 1, 2])
