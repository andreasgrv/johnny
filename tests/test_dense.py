import pytest
import numpy as np
from johnny.models import Dense
from chainer import optimizers, optimizer

def test_pred_dimensionality_basic():
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]
    oh_heads = [[1,0], [2,1,1,3],[0]]

    dm = Dense(10, 10, pos_units=1, word_units=1, lstm_units=1)
    r = dm(oh_words, oh_pos, oh_heads)
    assert(len(r[0]) == 2)
    assert(len(r[1]) == 4)
    assert(len(r[2]) == 1)

def test_pred_dimensionality_larger_hidden():
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]
    oh_heads = [[1,0], [2,1,1,3],[0]]

    dm = Dense(10, 10, pos_units=1, word_units=1, lstm_units=4)
    r = dm(oh_words, oh_pos, oh_heads)
    print(r)
    assert(len(r[0]) == 2)
    assert(len(r[1]) == 4)
    assert(len(r[2]) == 1)

def test_pred_dimensionality_wrong_input():
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]
    oh_heads = [[1,0], [2,1,1,3]]

    dm = Dense(10, 10, pos_units=1, word_units=1, lstm_units=1)
    with pytest.raises(AssertionError):
        r = dm(oh_words, oh_pos, oh_heads)

    oh_heads = [[1,0], [2,1,1,3], [0, 1]]
    with pytest.raises(Exception):
        r = dm(oh_words, oh_pos, oh_heads)

def test_can_predict_correct():
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]

    opt = optimizers.Adam(alpha=0.1)
    np.random.seed(13)

    model = Dense(10, 10, pos_units=1, word_units=1, lstm_units=8)
    opt.setup(model)
    # gradient clipping
    opt.add_hook(optimizer.GradientClipping(threshold=5))

    oh_heads = [[1,0], [2,1,1,3], [0]]
    loss = 1000.
    for i in range(50):
        r = model(oh_words, oh_pos, oh_heads)
        new_loss = model.loss.data
        assert(new_loss < loss)
        model.cleargrads()
        model.loss.backward()
        model.reset_state()
        # update parameters
        opt.update()
    assert(r == oh_heads)
