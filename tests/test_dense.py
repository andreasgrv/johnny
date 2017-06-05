import pytest
import chainer
import numpy as np
from johnny.models import Dense
from chainer import optimizers, optimizer

np.random.seed(13)

def test_pred_dimensionality_basic():
    # root is 9
    oh_words = [[9, 1,2], [9, 1,2,3,4], [9, 3]]
    oh_pos = [[9, 5,6], [9, 5,6,7,8], [9, 1]]
    oh_heads = [[1,0], [2,1,1,3],[0]]

    dm = Dense(10, 10, pos_units=1, word_units=1, lstm_units=1)
    r = dm(oh_words, oh_pos, oh_heads)
    assert(len(r[0]) == 2)
    assert(len(r[1]) == 4)
    assert(len(r[2]) == 1)

def test_pred_dimensionality_larger_hidden():
    oh_words = [[9, 1,2], [9, 1,2,3,4], [9, 3]]
    oh_pos = [[9, 5,6], [9, 5,6,7,8], [9, 1]]
    oh_heads = [[1,0], [2,1,1,3],[0]]

    dm = Dense(10, 10, pos_units=1, word_units=1, lstm_units=4)
    r = dm(oh_words, oh_pos, oh_heads)
    print(r)
    assert(len(r[0]) == 2)
    assert(len(r[1]) == 4)
    assert(len(r[2]) == 1)

def test_pred_dimensionality_wrong_input():
    oh_words = [[9, 1,2], [9, 1,2,3,4], [9, 3]]
    oh_pos = [[9, 5,6], [9, 5,6,7,8], [9, 1]]
    oh_heads = [[1,0], [2,1,1,3]]

    dm = Dense(10, 10, pos_units=1, word_units=1, lstm_units=1)
    with pytest.raises(AssertionError):
        r = dm(oh_words, oh_pos, oh_heads)

    oh_heads = [[1,0], [2,1,1,3], [0, 1]]
    with pytest.raises(Exception):
        r = dm(oh_words, oh_pos, oh_heads)

def test_can_predict_correct():
    oh_words = [[9, 1,2], [9, 1,2,3,4], [9, 3]]
    oh_pos = [[9, 5,6], [9, 5,6,7,8], [9, 1]]

    opt = optimizers.Adam(alpha=0.1)

    model = Dense(10, 10, pos_units=1, word_units=1, lstm_units=8)
    opt.setup(model)
    # gradient clipping
    opt.add_hook(optimizer.GradientClipping(threshold=5))

    oh_heads = [[1,0], [2,1,1,3], [0]]
    loss = 1000.
    for i in range(30):
        r = model(oh_words, oh_pos, oh_heads)
        new_loss = model.loss.data
        assert(new_loss < loss)
        model.cleargrads()
        model.loss.backward()
        model.reset_state()
        # update parameters
        opt.update()
    assert(r == oh_heads)

def test_batching_same_size():
    model = Dense(10, 10, pos_units=10, word_units=10, lstm_units=8, dropout_inp=0, dropout_rec=0)

    oh_words = [[9,1,2], [9,2,1], [9,3,2]]
    oh_pos = [[9,5,6], [9,1,3], [9,2,2]]
    oh_heads = [[1,0],[0,1], [1,0]]

    r = model(oh_words, oh_pos, oh_heads)
    batch_attn = model.attn.data[:]
    oh_words = [[9,1,2]]
    oh_pos = [[9,5,6]]
    oh_heads = [[1,0]]
    model.reset_state()
    r = model(oh_words, oh_pos)
    single_attn = model.attn.data
    assert(np.allclose(batch_attn[0], single_attn))

def test_batching_diff_size():
    model = Dense(10, 10, pos_units=10, word_units=10, lstm_units=8, dropout_inp=0, dropout_rec=0)

    oh_words = [[9,1,2], [9,1,2,3,4], [9,3]]
    oh_pos = [[9,5,6], [9,5,6,7,8], [9,1]]
    oh_heads = [[1,0], [2,1,1,3],[0]]

    r = model(oh_words, oh_pos, oh_heads)
    batch_attn = model.attn.data[:]
    oh_words = [[9,1,2]]
    oh_pos = [[9,5,6]]
    oh_heads = [[1,0]]
    model.reset_state()
    r = model(oh_words, oh_pos)
    single_attn = model.attn.data
    # NOTE: because the max length size is 4, the attention matrix is
    # squarish (not square cause of root) and padded
    assert(np.allclose(batch_attn[0,:3, :2], single_attn))
