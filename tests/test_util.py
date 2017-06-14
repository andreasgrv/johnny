import os
import numpy as np
import pytest
from johnny.utils import BucketManager, Experiment
from johnny import EXP_ENV_VAR

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

def test_right_leak_exact():
    data = [[[1,2,3,4],[1,2,3,4]],
            [[1,2],[1,2]]]

    batch_size = 2 
    bm = BucketManager(data, 1, 5, right_leak=2, batch_size=batch_size, shuffle=False, row_key=lambda x: len(x[0]))
    # make it improbable that it will select bucket with index 4 instead of 1
    bm.left_samples[1] = 1000
    batch = next(bm)
    assert(len(batch) == 2)
    assert([[1,2],[1,2]] in batch)
    assert([[1,2,3,4],[1,2,3,4]] in batch)

def test_right_leak_minus_one():
    data = [[[1,2,3,4],[1,2,3,4]],
            [[1,2],[1,2]]]

    batch_size = 2 
    bm = BucketManager(data, 1, 5, right_leak=1, batch_size=batch_size, shuffle=False, row_key=lambda x: len(x[0]))
    # make it improbable that it will select bucket with index 4 instead of 1
    bm.left_samples[1] = 1000
    batch = next(bm)
    assert(len(batch) == 1)
    assert([[1,2,3,4],[1,2,3,4]] not in batch)

def test_right_leak_overzealous():
    data = [[[1,2,3,4],[1,2,3,4]],
            [[1,2,3], [1,2,3]],
            [[1,2],[1,2]]]

    batch_size = 2 
    bm = BucketManager(data, 1, 5, right_leak=100, batch_size=batch_size, shuffle=False, row_key=lambda x: len(x[0]))
    # make it improbable that it will select bucket with index 4 instead of 1
    bm.left_samples[1] = 1000
    batch = next(bm)
    assert(len(batch) == 2)
    assert([[1,2],[1,2]] in batch)
    assert([[1,2,3],[1,2,3]] in batch)

def test_right_leak_right_one():
    data = [[[1,2,3,4],[1,2,3,4]],
            [[1,2],[1,2]]]

    batch_size = 2 
    bm = BucketManager(data, 1, 5, right_leak=100, batch_size=batch_size, shuffle=False, row_key=lambda x: len(x[0]))
    # make it improbable that it will select bucket with index 4 instead of 1
    bm.left_samples[1] = 1000
    batch = next(bm)
    assert(len(batch) == 2)
    assert([[1,2],[1,2]] in batch)
    assert([[1,2,3,4],[1,2,3,4]] in batch)

def test_experiment_to_and_from_yaml(tmpdir):
    p = str(tmpdir.mkdir('exps'))
    e = Experiment('test', lang='English', exp_folder_path=p, model_params={'lr': 0.5, 'lstm_units': 100})
    yml = e.to_yaml()
    assert(yml == Experiment.from_yaml(e.to_yaml()).to_yaml())

def test_experiment_load_save(tmpdir):
    p = str(tmpdir.mkdir('exps'))
    e = Experiment('test', lang='English', exp_folder_path=p, model_params={'lr': 0.5, 'lstm_units': 100})
    e.save(exp_folder_path=p)
    yml = e.to_yaml()
    e2 = Experiment.load(e.filepath)
    assert(yml == e2.to_yaml())

def test_experiment_check_os_environ(tmpdir):
    p = str(tmpdir.mkdir('exps'))
    os.environ[EXP_ENV_VAR] = p
    e = Experiment('test', lang='English', model_params={'lr': 0.5, 'lstm_units': 100})
    e.save()
    os.environ[EXP_ENV_VAR] = ''
    with pytest.raises(ValueError):
        e = Experiment('test', lang='English', model_params={'lr': 0.5, 'lstm_units': 100})
        e.save()

def test_experiment_non_existant_file(tmpdir):
    os.environ[EXP_ENV_VAR] = 'gobbledygook'
    with pytest.raises(ValueError):
        e = Experiment('test', lang='English', model_params={'lr': 0.5, 'lstm_units': 100})
        e.save()
