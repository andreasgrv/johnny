from johnny.vocab import Vocab

def test_from_token_list():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab.from_token_list(s.split(), size=7)
    assert(len(v.index) == len(set(s.split())))
    e = v.encode('unknown words'.split())
    assert(e == (0, 0))
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] != v.reserved.UNK)
    assert(e[1] != v.reserved.UNK)
    assert(e[2] == v.reserved.UNK)

def test_zero_size():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab.from_token_list(s.split(), size=0)
    assert(len(v.index) == 0)
    e = v.encode('unknown words'.split())
    assert(e == (0, 0))
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] == v.reserved.UNK)
    assert(e[1] == v.reserved.UNK)
    assert(e[2] == v.reserved.UNK)

def test_threshold():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab.from_token_list(s.split(), size=7, threshold=1)
    assert(len(v.index) == 1)
    e = v.encode('unknown words'.split())
    assert(e == (0, 0))
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] != v.reserved.UNK)
    assert(e[1] == v.reserved.UNK)
    assert(e[2] == v.reserved.UNK)

def test_serialisation(tmpdir):
    f = tmpdir.join('v.vocab')
    s = ['here', 'i', 'go', 'playing', 'the', 'fool', 'again', 'yes', 'i', 'am', 'i', 'am', 'i', 'am']
    v = Vocab.from_token_list(s, size=20)
    v.save(str(f))
    v2 = Vocab.load(str(f))
    for w in s:
        assert v[w] == v2[w]

def test_start_end_padding():
    s = "See my dreams they're not like anyone's"
    v = Vocab.from_token_list(s.split(), size=7)
    e = v.encode('See my dreams'.split(), with_start=True)
    assert(len(e) == 4)
    assert(e[0] == v.reserved.START)
    e = v.encode('See my dreams'.split(), with_end=True)
    assert(len(e) == 4)
    assert(e[0] != v.reserved.START)
    assert(e[-1] == v.reserved.END)
    e = v.encode('See my dreams'.split(), with_start=True, with_end=True)
    assert(len(e) == 5)
    assert(e[0] == v.reserved.START)
    assert(e[-1] == v.reserved.END)
