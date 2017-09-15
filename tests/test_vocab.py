from johnny.vocab import Vocab

def test_fit():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab(size=7).fit(s.split())
    assert(len(v.index) == len(set(s.split())))
    e = v.encode('unknown words'.split())
    assert(e == (v.reserved.UNK, v.reserved.UNK))
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] != v.reserved.UNK)
    assert(e[1] != v.reserved.UNK)
    assert(e[2] == v.reserved.UNK)

def test_zero_size():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab(size=0).fit(s.split())
    assert(len(v.index) == 0)
    e = v.encode('unknown words'.split())
    assert(e == (v.reserved.UNK, v.reserved.UNK))
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] == v.reserved.UNK)
    assert(e[1] == v.reserved.UNK)
    assert(e[2] == v.reserved.UNK)

def test_threshold():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab(size=7, threshold=1).fit(s.split())
    assert(len(v.index) == 1)
    e = v.encode('unknown words'.split())
    assert(e == (v.reserved.UNK, v.reserved.UNK))
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] != v.reserved.UNK)
    assert(e[1] == v.reserved.UNK)
    assert(e[2] == v.reserved.UNK)

def test_serialisation(tmpdir):
    f = tmpdir.join('v.vocab')
    s = ['here', 'i', 'go', 'playing', 'the', 'fool', 'again', 'yes', 'i', 'am', 'i', 'am', 'i', 'am']
    v = Vocab(size=20).fit(s)
    v.save(str(f))
    v2 = Vocab.load(str(f))
    for w in s:
        assert v[w] == v2[w]
