import pytest
import chainer
import numpy as np
from johnny.models import GraphParser
from johnny.components import Embedder, SentenceEncoder
from chainer import optimizers

SEED = 15

@pytest.fixture
def simple_word_model():
    np.random.seed(SEED)
    embed = Embedder((10,), (10,), dropout=0.)
    encoder = SentenceEncoder(embed, num_units=8, dropout=0.)
    model = GraphParser(encoder, mlp_arc_units=8, mlp_lbl_units=8, lbl_dropout=0.,
            arc_dropout=0., treeify='none', debug=True)
    return model

@pytest.fixture
def simple_pos_model():
    np.random.seed(SEED)
    embed = Embedder((10, 10), (10, 10), dropout=0.)
    encoder = SentenceEncoder(embed, num_units=8, dropout=0.)
    model = GraphParser(encoder, mlp_arc_units=8, mlp_lbl_units=8, lbl_dropout=0.,
            arc_dropout=0., treeify='none', debug=True)
    return model

@pytest.fixture
def dropout_pos_model():
    def instance():
        np.random.seed(SEED)
        embed = Embedder((10, 10), (10, 10), dropout=0.2)
        encoder = SentenceEncoder(embed, num_units=8, dropout=0.2)
        return GraphParser(encoder, mlp_arc_units=8, mlp_lbl_units=8, lbl_dropout=0.,
                arc_dropout=0., treeify='none', debug=True)
    return instance

def test_pred_dimensionality_basic(simple_word_model):
    # root is 9
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_heads = [[1,0], [2,1,1,3],[0]]
    oh_labels = [[1,0], [2,1,1,3],[0]]

    r, l = simple_word_model(oh_words, heads=oh_heads, labels=oh_labels)
    assert(len(r[0]) == 2)
    assert(len(r[1]) == 4)
    assert(len(r[2]) == 1)

def test_pred_dimensionality_wrong_input(simple_word_model):
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_heads = [[1,0], [2,1,1,3]]
    oh_labels = [[1,0], [2,1,1,3],[0]]

    with pytest.raises(IndexError):
        r, l = simple_word_model(oh_words, heads=oh_heads, labels=oh_labels)

    # - If you don't give enough head training signal it's up to you
    # oh_heads = [[1,0], [2,1,1,3], [0, 1]]
    # with pytest.raises(Exception):
    #     r = dm(oh_words, oh_pos, oh_heads)

def test_can_predict_correct(simple_word_model):
    oh_words = [[6,7], [1,2,3,4], [5]]

    opt = optimizers.Adam(alpha=0.1)

    opt.setup(simple_word_model)

    oh_heads = [[1,0], [2,3,1,3], [0]]
    oh_labels = [[3,5], [2,4,5,3],[2]]
    loss = 1000.
    for i in range(10):
        r, l = simple_word_model(oh_words, heads=oh_heads, labels=oh_labels)
        new_loss = simple_word_model.loss.data
        assert(new_loss < loss)
        loss = new_loss
        simple_word_model.cleargrads()
        simple_word_model.loss.backward()
        # update parameters
        opt.update()
    assert([h.tolist() for h in r] == oh_heads)
    assert([e.tolist() for e in l] == oh_labels)

def test_batching_same_size_dropout(dropout_pos_model):

    oh_words = [[1,2], [2,1], [3,2]]
    oh_pos = [[5,6], [1,3], [2,2]]

    model = dropout_pos_model()
    with chainer.using_config('train', False):
        r, l = model(oh_words, oh_pos)
    batch_arcs = r[:]
    oh_words = [[1,2], [5,2], [4,3]]
    oh_pos = [[5,6], [3,5], [3,4]]
    model2 = dropout_pos_model()
    with chainer.using_config('train', False):
        r, l = model2(oh_words, oh_pos)
    single_arcs = r
    assert(np.allclose(batch_arcs[0], single_arcs[0]))

def test_batching_diff_size_dropout(dropout_pos_model):

    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]

    model = dropout_pos_model()
    with chainer.using_config('train', False):
        r, l = model(oh_words, oh_pos)
    batch_arcs = model.arcs

    oh_words = [[1,2], [5,2,3], [4]]
    oh_pos = [[5,6], [3,5,6], [3]]
    # reset for dropout - need same matrices
    np.random.seed(SEED)
    model2 = dropout_pos_model()
    with chainer.using_config('train', False):
        r, l = model2(oh_words, oh_pos)
    single_arcs = model2.arcs
    # NOTE: because the max length size is 4, the attention matrix is
    # squarish (not square cause of root) and padded
    assert(np.allclose(batch_arcs[0,:3, :2], single_arcs[0, :3, :2]))

def test_batching_same_size(simple_pos_model):

    oh_words = [[1,2], [2,1], [3,2]]
    oh_pos = [[5,6], [1,3], [2,2]]

    with chainer.using_config('train', False):
        r, l = simple_pos_model(oh_words, oh_pos)
    batch_arcs = r
    oh_words = [[2,1]]
    oh_pos = [[1,3]]
    with chainer.using_config('train', False):
        r, l = simple_pos_model(oh_words, oh_pos)
    single_arcs = r
    assert(np.allclose(batch_arcs[1], single_arcs))

def test_batching_diff_size(simple_pos_model):
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]

    with chainer.using_config('train', False):
        r, l = simple_pos_model(oh_words, oh_pos)
    batch_arcs = simple_pos_model.arcs
    oh_words = [[1,2]]
    oh_pos = [[5,6]]
    with chainer.using_config('train', False):
        r, l = simple_pos_model(oh_words, oh_pos)
    single_arcs = simple_pos_model.arcs
    # NOTE: because the max length size is 4, the attention matrix is
    # squarish (not square cause of root) and padded
    assert(np.allclose(batch_arcs[0,:3, :2], single_arcs))

def test_batched_preds_equal_non_batched_preds(simple_pos_model):
    oh_words = [[1,2], [1,2,3,4], [3]]
    oh_pos = [[5,6], [5,6,7,8], [1]]

    with chainer.using_config('train', False):

        b_preds, l_p = simple_pos_model(oh_words, oh_pos)

        for i, (w, p) in enumerate(zip(oh_words, oh_pos)):
            pred, l = simple_pos_model([w], [p])
            assert(np.allclose(pred, b_preds[i]))
