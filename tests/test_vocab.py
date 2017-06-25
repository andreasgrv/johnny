from johnny.dep import Vocab

def test_from_token_list():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab.from_token_list(s.split(), size=7)
    assert(len(v) == len(set(s.split())))
    e = v.encode('unknown words'.split())
    assert(e == [0, 0])
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] != v.UNK)
    assert(e[1] != v.UNK)
    assert(e[2] == v.UNK)

def test_zero_size():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab.from_token_list(s.split(), size=0)
    assert(len(v) == 0)
    e = v.encode('unknown words'.split())
    assert(e == [0, 0])
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] == v.UNK)
    assert(e[1] == v.UNK)
    assert(e[2] == v.UNK)

def test_threshold():
    s = 'daybreak at the bottom of the lake' # note there are 2 "the"
    v = Vocab.from_token_list(s.split(), size=7, threshold=1)
    print(v.index)
    assert(len(v) == 1)
    e = v.encode('unknown words'.split())
    assert(e == [0, 0])
    e = v.encode('the daybreak supercalifragilistic'.split())
    assert(e[0] != v.UNK)
    assert(e[1] == v.UNK)
    assert(e[2] == v.UNK)

def test_serialisation(tmpdir):
    f = tmpdir.join('v.vocab')
    s = ['here', 'i', 'go', 'playing', 'the', 'fool', 'again', 'yes', 'i', 'am', 'i', 'am', 'i', 'am']
    v = Vocab.from_token_list(s, size=20)
    v.save(str(f))
    v2 = Vocab.load(str(f))
    for w in s:
        assert v[w] == v2[w]
