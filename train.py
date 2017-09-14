from __future__ import division
import os
import sys
import numpy as np
import chainer
import pickle
from mlconf import YAMLLoaderAction, ArgumentParser
from tqdm import tqdm
from itertools import chain
from johnny import EXP_ENV_VAR
from johnny.dep import UDepLoader
from johnny.vocab import Vocab, AbstractVocab, UDepVocab, UPOSVocab
from johnny.utils import BucketManager
from johnny.metrics import Average, UAS, LAS
import johnny.preprocess as pp


np.set_printoptions(precision=5, suppress=True)


def seed_chainer(seed, gpu_id):
    np.random.seed(seed)
    if gpu_id >= 0:
        chainer.config.cudnn_deterministic = True
        # TODO: ask chainer devs about this?
        # there seems to be no easy way to reset the seed!
        chainer.cuda.cupy.random.seed(seed)
        chainer.cuda.cupy.random.get_random_state()
        rs = chainer.functions.connection.n_step_rnn.DropoutRandomStates(seed)
        chainer.functions.connection.n_step_rnn._random_states[gpu_id] = rs



def preprocess(word, conf):
    if conf.lowercase:
        word = word.lower()
    # replace numbers with __NUM__
    if conf.collapse_nums:
        word = pp.collapse_nums(word)
    # collapse more than 3 repetitions of a character
    # to two repetitions TODO: maybe this is bad for some langs?
    if conf.collapse_triples:
        word = pp.collapse_triples(word)
    if conf.remove_diacritics:
        word = pp.remove_diacritics(word)
    if conf.expand_diacritics:
        word = pp.expand_diacritics(word)
    return word


def to_ngrams(word, n=1):
    assert(n > 0)
    if n == 1:
        return tuple(word)
    return tuple(word[i:i+n] for i in range(len(word)-n+1))


def create_vocabs(t_set, conf):
    # we don't need to pad in this case
    if conf.ngram <= 0:
        t_tokens = ((preprocess(w, conf.preprocess) for w in s)
                    for s in t_set.words)
    else:
        t_tokens = (chain.from_iterable(to_ngrams(preprocess(w, conf.preprocess), n=conf.ngram)
                                        for w in s)
                    for s in t_set.words)

    v_word = Vocab.from_token_list(chain.from_iterable(t_tokens),
                                   out_size=conf.vocab.size,
                                   threshold=conf.vocab.threshold)

    # if we are using the CONLL2017 dataset (universal dependencies)
    # then we know the vocabulary beforehand. We use the full vocabulary
    # with predefined keys because it is less errorprone, and because we
    # know the labels of the indices for the confusion matrix.
    if 'v2_0' in conf.dataset.name:
        v_pos = UPOSVocab()
        v_arcs = UDepVocab()
    else:
        v_pos = AbstractVocab()
        v_arcs = AbstractVocab(with_reserved=False)
    vocabs = (v_word, v_pos, v_arcs)
    return vocabs


def data_to_rows(data, vocabs, conf):
    v, vpos, varcs = vocabs
    if conf.ngram <= 0:
        words_indices = tuple(v.encode(preprocess(w, conf.preprocess)
                               for w in s)
                               for s in data.words)
    else:
        words_indices = tuple(tuple(v.encode(to_ngrams(preprocess(w, conf.preprocess), n=conf.ngram))
                                             for w in s)
                              for s in data.words)
    pos_indices = tuple(map(vpos.encode, data.upostags))
    labels_indices = tuple(map(varcs.encode, data.arctags))
    heads = data.heads
    data_rows = zip(words_indices, pos_indices, heads, labels_indices)
    return tuple(data_rows)


def to_batches(rows, batch_size, sort=False):
    if sort:
        rows = sorted(rows, key=lambda x:len(x[0]), reverse=True)
    i = 0
    batch = rows[i: i + batch_size]
    while(batch):
        yield batch
        i += batch_size
        batch = rows[i: i + batch_size]


def visualise_dict(d, num_items=50):
    buff = []
    window_width = os.get_terminal_size().columns
    widths = (15, 2, 5)
    lentry_width, pad, rentry_width = widths
    entry_width = sum(widths)
    per_line = window_width//entry_width
    fmt = ('{w:%d.%d}%s{i:%d.%d}'
            % (lentry_width, lentry_width, ' '*pad, rentry_width, rentry_width))
    for i, (key, val) in enumerate(d.items()):
        buff.append((key, val))
        if len(buff) == per_line:
            print(' '.join((fmt.format(w=w, i=str(i)) for w, i in buff)))
            buff = []
        if i > num_items:
            break
    print(' '.join((fmt.format(w=w, i=str(i)) for w, i in buff)))
    print('\n\n')


def train_epoch(model, optimizer, buckets, data_size):
    iters = 0
    tf_str = 'Train: batch_size={0:d}, mean loss={1:.2f}, mean UAS={2:.3f} mean LAS={3:.3f}'
    with tqdm(total=data_size, leave=False) as pbar, \
        chainer.using_config('train', True):

        mean_loss = Average()
        u_scorer = UAS()
        l_scorer = LAS()
        for batch in buckets:
            seqs = list(zip(*batch))
            label_batch = seqs.pop()
            head_batch = seqs.pop()
            arc_preds, lbl_preds = model(*seqs, heads=head_batch, labels=label_batch)
            loss = model.loss
            model.cleargrads()
            loss.backward()
            optimizer.update()

            loss_value = float(loss.data)

            for p_arcs, p_lbls, t_arcs, t_lbls in zip(arc_preds, lbl_preds, head_batch, label_batch):
                u_scorer(arcs=(p_arcs, t_arcs))
                l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))
            mean_loss(loss_value)
            out_str = tf_str.format(len(batch), mean_loss.score, u_scorer.score, l_scorer.score)
            pbar.set_description(out_str)
            iters += len(batch)
            pbar.update(len(batch))
            if iters >= data_size:
                break
        time_taken = pbar._time() - pbar.start_t
    stats = {'train_time': time_taken,
             'train_mean_loss': mean_loss.score,
             'train_uas': u_scorer.score,
             'train_las': l_scorer.score}
    return stats


def eval_epoch(model, buckets, data_size, label='', num_labels=None):
    def label_stat(stat):
        return '%s_%s' % (label, stat)

    tf_str = ('Eval - %s : batch_size={0:d}, mean loss={1:.2f}, '
              'mean UAS={2:.3f} mean LAS={3:.3f}' % label)
    with tqdm(total=data_size, leave=False) as pbar, \
        chainer.using_config('train', False), \
        chainer.no_backprop_mode():

        mean_loss = Average()
        u_scorer = UAS()
        l_scorer = LAS(num_labels=num_labels)
        for batch in buckets:
            # model.reset_state()
            seqs = list(zip(*batch))
            label_batch = seqs.pop()
            head_batch = seqs.pop()
            arc_preds, lbl_preds = model(*seqs, heads=head_batch, labels=label_batch)
            loss = model.loss

            loss_value = float(loss.data)

            for p_arcs, p_lbls, t_arcs, t_lbls in zip(arc_preds, lbl_preds, head_batch, label_batch):
                u_scorer(arcs=(p_arcs, t_arcs))
                l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))
            mean_loss(loss_value)
            out_str = tf_str.format(len(batch), mean_loss.score, u_scorer.score, l_scorer.score)
            pbar.set_description(out_str)
            pbar.update(len(batch))
    # if num_labels is None:
    #     conf_matrix = [[]]
    # else:
    #     conf_matrix = l_scorer.conf_matrix.tolist()
    # conf_matrix = [[]]
    stats = {label_stat('mean_loss'): mean_loss.score,
             label_stat('uas'): u_scorer.score,
             label_stat('las'): l_scorer.score}
             # label_stat('conf_matrix'): conf_matrix}
    return stats


def train_loop(train_rows, dev_rows, conf, checkpoint_callback=None, gpu_id=-1):

    train_buckets = BucketManager(train_rows,
                                  conf.train_buckets.bucket_width,
                                  conf.dataset.train_max_sent_len,
                                  shuffle=True,
                                  batch_size=conf.batch_size,
                                  right_leak=conf.train_buckets.right_leak,
                                  row_key=lambda x: len(x[0]),
                                  loop_forever=True)
    dev_batches = tuple(to_batches(dev_rows, conf.dev_batch_size, sort=True))

    print('training max seq len ', train_buckets.max_len)

    model = conf.model
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    opt = chainer.optimizers.Adam(alpha=conf.optimizer.learning_rate)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(threshold=conf.optimizer.grad_clip))

    e = 0
    best_valid_las = 0.
    patience = conf.checkpoint.patience
    # checkpoint.every defines how often to checkpoint in multiples of
    # the batch size.  if conf.every is <= 0 then we checkpoint each epoch
    cp_iters = conf.batch_size * conf.checkpoint.every \
        if conf.checkpoint.every > 0 else len(train_rows)
    iters_per_epoch = len(train_rows)
    current_iters = 0
    current_checkpoint = 0

    pbar = tqdm(desc='Epoch 0 - Patience %d' % patience)
    while e < conf.max_epochs:
        checkpoint_stats = dict()
        # train
        stats = train_epoch(model, opt, train_buckets, cp_iters)

        checkpoint_stats.update(**stats)

        # score dev set
        stats = eval_epoch(model, dev_batches, data_size=len(dev_rows),
                           label='valid', num_labels=conf.model.num_labels)
        checkpoint_stats.update(**stats)

        if checkpoint_stats['valid_las'] > best_valid_las:
            best_valid_las = checkpoint_stats['valid_las']
            patience = conf.checkpoint.patience
        else:
            patience -= 1
        checkpoint_stats.update(patience=patience)

        current_iters += cp_iters
        e = int(current_iters / iters_per_epoch)
        current_checkpoint += 1
        pbar.set_description('Epoch %d - Patience %d - Best LAS: %.2f UAS: %.2f'
                             % (e, patience, best_valid_las * 100,
                                checkpoint_stats['valid_uas'] * 100))
        pbar.update()

        if checkpoint_callback is not None:
            checkpoint_callback(e, checkpoint_stats,
                                improved=(patience == conf.checkpoint.patience))

        if patience == 0:
            break
    pbar.close()
    return model

if __name__ == "__main__":

    parser = ArgumentParser(description='Dependency parser trainer')

    if not EXP_ENV_VAR in os.environ:
        parser.add_argument('-o', '--outfolder', required=True, type=str,
                            help='path to where to save the models.')
    parser.add_argument('-i', '--datafolder', required=False, type=str,
                        help='path to CONLL folder containing languages. '
                        'If not set script will check env variables.')
    parser.add_argument('--name', type=str, default='test-model',
                        help='What to name the experiment.')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='Which gpu device to use, -1 means cpu.')
    parser.add_argument('--visualise', action='store_true',
                        help='Whether to visualise training or not.')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to print additional info.')
    parser.add_argument('--load_blueprint', action=YAMLLoaderAction)

    conf = parser.parse_args()

    outfolder = conf.get('outfolder', os.environ.get(EXP_ENV_VAR))

    # setup seeds for reproducibility 
    seed_chainer(conf.seed, conf.gpu_id)

    if conf.verbose:
        print('Loaded Blueprint settings:\n%s\n' % conf)

    print('Loading dataset...')
    udep = UDepLoader(conf.dataset.name, datafolder=conf.datafolder)
    t_set, v_set = udep.load_train_dev(conf.dataset.lang, verbose=conf.verbose)

    conf.dataset.train_max_sent_len = t_set.len_stats['max_sent_len']
    conf.dataset.dev_max_sent_len = v_set.len_stats['max_sent_len']

    vocabs = create_vocabs(t_set, conf)
    v_word, v_pos, v_arc = vocabs

    train_rows = data_to_rows(t_set, vocabs, conf)
    dev_rows = data_to_rows(v_set, vocabs, conf)

    if conf.verbose:
        for v in vocabs:
            print(v)
            visualise_dict(v.index, num_items=50)

    if conf.ngram <= 0:
        conf.model.encoder.embedder.in_sizes = [len(v_word), len(v_pos)]
    else:
        conf.model.encoder.embedder.word_encoder.vocab_size = len(v_word)
        conf.model.encoder.embedder.in_sizes = [len(v_pos)]
    conf.model.num_labels = len(v_arc)

    # built_conf has all class representations instantiated
    # we need this here because otherwise we wouldn't be able to set random seed
    # or modify input sizes according to vocabsize dynamically
    # since we don't know the sizes when we create the blueprint
    built_conf = conf.build(verbose=conf.verbose)

    # ================ Save model ======================
    # chainer.serializers.save_npz('testme', model)
    # timestamp = datetime.datetime.strftime(datetime.datetime.now(),
    #                                        '%d-%m-%Y+%H:%M:%S')
    # filename = '%s@%s' % (conf.name, timestamp)
    filename = conf.name
    blueprint_filename = '%s.bp' % filename
    model_filename = '%s.model' % filename
    vocab_filename = '%s.vocab' % filename
    dataset_folder = os.path.join(outfolder, conf.dataset.name.lower())
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    lang_folder = os.path.join(dataset_folder, conf.dataset.lang.lower())
    if not os.path.isdir(lang_folder):
        os.mkdir(lang_folder)
    blueprint_path = os.path.join(lang_folder, blueprint_filename)
    model_path = os.path.join(lang_folder, model_filename)
    vocab_path = os.path.join(lang_folder, vocab_filename)

    # prepare for results
    conf.results = dict()

    def on_epoch_end(epoch, epoch_stats, improved):
        if conf.results:
            for key, value in epoch_stats.items():
                conf.results[key].append(value)
        else:
            for key, value in epoch_stats.items():
                conf.results[key] = [value]
        if improved:
            print(' Saving model..')
            chainer.serializers.save_npz(model_path, built_conf.model)

    if conf.visualise:
        import pynput

        def on_press(key):
            INCREMENT = 0.1
            if key == pynput.keyboard.Key.esc:
                built_conf.model.visualise = not built_conf.model.visualise
            elif key == pynput.keyboard.Key.up:
                built_conf.model.sleep_time += INCREMENT
            elif key == pynput.keyboard.Key.down:
                if built_conf.model.sleep_time >= INCREMENT:
                    built_conf.model.sleep_time -= INCREMENT

        if 'v2_0' not in built_conf.dataset.name:
            print('### Sorry! visualisation only supported for Universal Dependencies v2.0\n'
                  'Try without the --visualise flag.')
            sys.exit(1)
        try:
            with pynput.keyboard.Listener(on_press=on_press) as listener:
                built_conf.model.visualise = True
                model = train_loop(train_rows, dev_rows, built_conf,
                                   checkpoint_callback=on_epoch_end,
                                   gpu_id=built_conf.gpu_id)
                pynput.keyboard.Listener.stop
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('Cannot use visualisation - try without')
    else:
        model = train_loop(train_rows, dev_rows, built_conf,
                           checkpoint_callback=on_epoch_end, gpu_id=built_conf.gpu_id)
    
    try:
        conf.model_path = model_path
        print('Writing vocabs to %s' % vocab_path)
        with open(vocab_path, 'wb') as pf:
            pickle.dump(vocabs, pf)
        conf.vocab_path = vocab_path
        print('Writing blueprint to %s' % blueprint_path)
        conf.to_file(blueprint_path)
    except Exception:
        os.remove(model_path)
        os.remove(vocab_path)
        os.remove(blueprint_path)
