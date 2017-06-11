import numpy as np
import pytest
from johnny.utils import BucketManager

def test_basic():
    data = [[1,2,3],
            [4,5,6,7],
            [8,9,3,2],
            [4,5,6],
            [3,4,5,6],
            [1,2,3]]

    batch_size = 3 
    bm = BucketManager(data, 1, 4, batch_size=batch_size)
    for each in bm:
        assert len(each) == batch_size

def test_uneven():
    data = [[1,2,3],
            [4,5,6],
            [1,2,3]]

    batch_size = 2 
    bm = BucketManager(data, 1, 3, batch_size=batch_size)
    batch = next(bm)
    assert(len(batch) == 2)
    batch = next(bm)
    assert(len(batch) == 1)
    with pytest.raises(StopIteration):
        batch = next(bm)

def test_wrong_max_len():
    data = [[1,2,3],
            [4,5,6],
            [1,2,3]]

    batch_size = 2 
    with pytest.raises(IndexError):
        bm = BucketManager(data, 1, 2, batch_size=batch_size)

def test_exact_max_len():
    data = [[1,2,3],
            [4,5,6],
            [1,2,3]]

    batch_size = 2 
    bm = BucketManager(data, 2, 3, batch_size=batch_size)
    assert(bm.buckets[1][bm.END_INDEX_KEY] == 3)

def test_bucket_width():
    data = [[1,2,3],
            [4,5,6],
            [1,2,3],
            [2,3],
            [5,2],
            ]

    bucket_width = 3
    bm = BucketManager(data, bucket_width, 3, batch_size=2)
    assert(bm.buckets[0][bm.END_INDEX_KEY] == 5)
    bucket_width = 2
    bm = BucketManager(data, bucket_width, 3, batch_size=2)
    assert(bm.buckets[0][bm.END_INDEX_KEY] == 2)
    assert(bm.buckets[1][bm.END_INDEX_KEY] == 3)

def test_nonzero_start():
    data = [[1,2,3],
            [4,5,6],
            [1,2,3],
            [2,3],
            [5,2],
            ]

    bucket_width = 2
    bm = BucketManager(data, bucket_width, 3, batch_size=2)
    assert(bm.buckets[0][bm.END_INDEX_KEY] == 2)
    assert(bm.buckets[1][bm.END_INDEX_KEY] == 3)

    bucket_width = 2
    bm = BucketManager(data, bucket_width, 4, min_len=2, batch_size=2)
    assert(bm.buckets[0][bm.END_INDEX_KEY] == 5)

def test_zero_length():
    data = [[],
            [],
            [],
            [2,3],
            [5,2],
            ]

    bucket_width = 2
    with pytest.raises(IndexError):
        bm = BucketManager(data, bucket_width, 2, batch_size=2)
    bm = BucketManager(data, bucket_width, 2, min_len=0, batch_size=2)
    assert(bm.buckets[0][bm.END_INDEX_KEY] == 3)
    assert(bm.buckets[1][bm.END_INDEX_KEY] == 2)
    bucket_width = 3
    bm = BucketManager(data, bucket_width, 2, min_len=0, batch_size=2)
    assert(bm.buckets[0][bm.END_INDEX_KEY] == 5)

def test_uneven_unshuffled():
    data = [[1,2,3],
            [4,5,6],
            [7,8,9]]

    batch_size = 2 
    bm = BucketManager(data, 1, 3, batch_size=batch_size, shuffle=False)
    batch = next(bm)
    assert(batch == [[1,2,3],[4,5,6]])
    batch = next(bm)
    assert(batch == [[7,8,9]])
    with pytest.raises(StopIteration):
        batch = next(bm)

def test_reset():
    data = [[1,2,3],
            [4,5,6]]

    batch_size = 2 
    bm = BucketManager(data, 1, 3, batch_size=batch_size, shuffle=False)
    batch = next(bm)
    assert(batch == [[1,2,3],[4,5,6]])
    with pytest.raises(StopIteration):
        batch = next(bm)
    bm.reset()
    batch = next(bm)
    assert(batch == [[1,2,3],[4,5,6]])
    with pytest.raises(StopIteration):
        batch = next(bm)


def test_row_key():
    data = [[[1,2,3],[1,2,3]],
            [[1,2],[1,2]],
            [[3,4,5],[3,4,5]],
            [[3,4],[3,4]],
            [[6,7,8],[6,7,8]]
            ]

    batch_size = 2 
    bm = BucketManager(data, 1, 3, batch_size=batch_size, shuffle=False, row_key=lambda x: len(x[0]))
    assert(bm.buckets[0][bm.DATA_KEY] == [])
    assert(bm.buckets[1][bm.END_INDEX_KEY] == 2)
    assert(bm.buckets[2][bm.END_INDEX_KEY] == 3)
    batch = next(bm)
    assert(len(batch) == 2)
    batch = next(bm)
    batch = next(bm)
    with pytest.raises(StopIteration):
        batch = next(bm)
    bm = BucketManager(data, 1, 3, batch_size=batch_size, shuffle=False, row_key=lambda x: len(x))
    assert(bm.buckets[0][bm.DATA_KEY] == [])
    assert(bm.buckets[1][bm.END_INDEX_KEY] == 5)
    assert(bm.buckets[2][bm.END_INDEX_KEY] == 0)
