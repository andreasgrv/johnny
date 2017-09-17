import chainer
import pickle
from tqdm import tqdm
from johnny.dep import UDepLoader
from johnny.metrics import Average, UAS, LAS
from train import visualise_dict, data_to_rows, to_batches
from mlconf import ArgumentParser, Blueprint


def test_loop(bp, test_set):

    model_path = bp.model_path
    vocab_path = bp.vocab_path

    with open(vocab_path, 'rb') as pf:
        vocabs = pickle.load(pf)

    # (v_word, v_pos, v_arcs) = vocabs
    (v_word, v_arcs) = vocabs
    visualise_dict(v_word.index, num_items=20)
    # visualise_dict(v_pos.index, num_items=20)
    visualise_dict(v_arcs.index, num_items=20)

    test_rows = data_to_rows(test_set, vocabs, bp)
    print(test_set)
    print(test_set[0][0])
    print(test_set[-1][-1])
    # Remove all info we are going to predict
    # to make sure we don't make a fool of ourselves
    # if we have a bug and gold data stays in its place
    test_set.unset_heads()
    test_set.unset_labels()
    test_set.unset_deps()
    test_set.unset_misc()
    print(test_set[0][0])
    print(test_set[-1][-1])
    print('test max seq len ', test_set.len_stats['max_sent_len'])

    built_bp = bp.build()
    model = built_bp.model
    chainer.serializers.load_npz(model_path, model)

    # test
    tf_str = ('Eval - test : batch_size={0:d}, mean loss={1:.2f}, '
              'mean UAS={2:.3f} mean LAS={3:.3f}')
    with tqdm(total=len(test_set)) as pbar, \
        chainer.using_config('train', False), \
        chainer.no_backprop_mode():

        mean_loss = Average()
        u_scorer = UAS()
        l_scorer = LAS()
        index = 0
        # NOTE: IMPORTANT!!
        # BATCH SIZE is important here to reproduce the results
        # for the cnn - since changing the batch size changes
        # has the effect of different words having different padding.
        # NOTE: test_mean_loss changes because it is averaged
        # across batches, so changing the number of batches affects it
        BATCH_SIZE = 256
        for batch in to_batches(test_rows, BATCH_SIZE, sort=False):
            batch_size = 0
            seqs = list(zip(*batch))
            label_batch = seqs.pop()
            head_batch = seqs.pop()
            arc_preds, lbl_preds = model(*seqs, heads=head_batch, labels=label_batch)
            loss = model.loss
            loss_value = float(loss.data)

            for p_arcs, p_lbls, t_arcs, t_lbls in zip(arc_preds, lbl_preds, head_batch, label_batch):
                u_scorer(arcs=(p_arcs, t_arcs))
                l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))
                # test_set[index].set_heads(p_arcs)
                # str_labels = (v_arcs.rev_index[l] for l in p_lbls)
                # test_set[index].set_labels(str_labels)
                index += 1
                batch_size += 1
            mean_loss(loss_value)
            out_str = tf_str.format(batch_size, mean_loss.score, u_scorer.score, l_scorer.score)
            pbar.set_description(out_str)
            pbar.update(batch_size)
    # make sure you aren't a dodo
    assert(index == len(test_set))

    stats = {'test_mean_loss': mean_loss.score,
             'test_uas': u_scorer.score,
             'test_las': l_scorer.score}

    # TODO: save these
    bp.test_results = stats
    for key, val in stats.items():
        print('%s: %s' % (key, val))


if __name__ == "__main__":
    # needed to import train to visualise_train
    parser = ArgumentParser(description='Dependency parser evaluator')
    parser.add_argument('--blueprint', required=True, type=str,
                        help='Path to .bp blueprint file produces by training.')
    parser.add_argument('--test_file', required=True, type=str,
                        help='Conll file to use for testing')
    parser.add_argument('--conll_out', action='store_true',
                        help='If specified writes conll output')
    parser.add_argument('--treeify', type=str, default='chu',
                        help='algorithm to postprocess arcs with. '
                        'Choose chu to allow for non projectivity, else eisner')

    args = parser.parse_args()

    CONLL_OUT = args.conll_out
    TREEIFY = args.treeify

    blueprint = Blueprint.from_file(args.blueprint)
    blueprint.model.treeify = TREEIFY

    test_data = UDepLoader.load_conllu(args.test_file)
    test_data.lang = blueprint.dataset.lang

    test_loop(blueprint, test_data)

    if CONLL_OUT:
        test_data.save(blueprint.model_path.replace('.model', '.conllu'))
