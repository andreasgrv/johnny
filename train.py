import os
import sys
import dill
import yaml
import chainer
import numpy as np
from mlconf import YAMLLoaderAction, ArgumentParser
from tqdm import tqdm
from itertools import chain
from collections import namedtuple, defaultdict
from johnny import EXP_ENV_VAR
from johnny.dep import UDepLoader
from johnny.vocab import Vocab, UDepVocab, UPOSVocab
from johnny.misc import visualise_dict, BucketManager
from johnny.metrics import Average, UAS, LAS
from johnny.text_utils import process_text, encode_texts


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


def dataset_to_cols(dataset, conf):
    data_tup = namedtuple('DataCols', ('text', 'heads', 'arcs', 'pos_tags'))
    text = tuple(process_text(s, conf.ngram, conf.subword, conf.preprocess)
                 for s in dataset.words)
    return data_tup(text, dataset.heads, dataset.arctags, dataset.upostags)


def data_to_rows(data, vocabs, conf):
    """Encodes input using vocabs where needed and returns a tuple of
    rows. Each row is a training instance containing both inputs and
    targets (inputs first, targets later)."""
    cols = []
    # add texts
    cols.append(encode_texts(data.text, vocabs.text, conf.subword))
    # add heads
    cols.append(data.heads)
    # add arc labels
    cols.append(tuple(map(vocabs.arcs.encode, data.arcs)))
    if conf.model.predict_pos:
        cols.append(tuple(map(vocabs.pos_tags.encode, data.pos_tags)))
    # NOTE: order of cols matters because we rely on it during unpacking.
    return tuple(zip(*cols))


def to_batches(rows, batch_size, sort=False):
    if sort:
        rows = sorted(rows, key=lambda x:len(x[0]), reverse=True)
    i = 0
    batch = rows[i: i + batch_size]
    while(batch):
        yield batch
        i += batch_size
        batch = rows[i: i + batch_size]


def train_epoch(model, optimizer, buckets, data_size):
    iters = 0
    tf_str = 'Train: batch_size={0:d}, mean loss={1:.2f}, mean LAS={2:.3f} mean UAS={3:.3f} mean POS={4:.3f}'
    with tqdm(total=data_size, leave=False) as pbar, \
        chainer.using_config('train', True):

        mean_loss = Average()
        u_scorer = UAS()
        l_scorer = LAS()
        t_scorer = UAS()
        for batch in buckets:
            seqs = list(zip(*batch))
            pos_tag_batch = seqs.pop() if model.predict_pos else None
            label_batch = seqs.pop()
            head_batch = seqs.pop()
            if model.predict_pos:
                arc_preds, lbl_preds, tag_preds = model(*seqs,
                                                        heads=head_batch,
                                                        labels=label_batch,
                                                        pos_tags=pos_tag_batch)

            else:
                arc_preds, lbl_preds = model(*seqs,
                                             heads=head_batch,
                                             labels=label_batch,
                                             pos_tags=pos_tag_batch)
            loss = model.loss
            model.cleargrads()
            loss.backward()
            optimizer.update()

            loss_value = float(loss.data)

            if model.predict_pos:
                for p_arcs, p_lbls, p_tags, t_arcs, t_lbls, t_tags \
                        in zip(arc_preds, lbl_preds, tag_preds, head_batch, label_batch, pos_tag_batch):
                    u_scorer(arcs=(p_arcs, t_arcs))
                    l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))
                    t_scorer(arcs=(p_tags, t_tags))
            else:
                for p_arcs, p_lbls, t_arcs, t_lbls in zip(arc_preds, lbl_preds, head_batch, label_batch):
                    u_scorer(arcs=(p_arcs, t_arcs))
                    l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))

            mean_loss(loss_value)
            out_str = tf_str.format(len(batch), mean_loss.score, l_scorer.score, u_scorer.score, t_scorer.score)
            pbar.set_description(out_str)
            iters += len(batch)
            pbar.update(len(batch))
            if iters >= data_size:
                break
        time_taken = pbar._time() - pbar.start_t
    stats = {'train_time': time_taken,
             'train_mean_loss': mean_loss.score,
             'train_uas': u_scorer.score,
             'train_las': l_scorer.score,
             'train_pos': t_scorer.score}
    return stats


def eval_epoch(model, buckets, data_size, label='', num_labels=None):
    def label_stat(stat):
        return '%s_%s' % (label, stat)

    tf_str = ('Eval - %s : batch_size={0:d}, mean loss={1:.2f}, '
              'mean UAS={2:.3f} mean LAS={3:.3f} mean POS={4:.3f}' % label)
    with tqdm(total=data_size, leave=False) as pbar, \
        chainer.using_config('train', False), \
        chainer.no_backprop_mode():

        mean_loss = Average()
        u_scorer = UAS()
        l_scorer = LAS(num_labels=num_labels)
        t_scorer = UAS()
        for batch in buckets:
            # model.reset_state()
            seqs = list(zip(*batch))
            pos_tag_batch = seqs.pop() if model.predict_pos else None
            label_batch = seqs.pop()
            head_batch = seqs.pop()
            if model.predict_pos:
                arc_preds, lbl_preds, tag_preds = model(*seqs,
                                                        heads=head_batch,
                                                        labels=label_batch,
                                                        pos_tags=pos_tag_batch)

            else:
                arc_preds, lbl_preds = model(*seqs,
                                             heads=head_batch,
                                             labels=label_batch,
                                             pos_tags=pos_tag_batch)
            loss = model.loss

            loss_value = float(loss.data)

            if model.predict_pos:
                for p_arcs, p_lbls, p_tags, t_arcs, t_lbls, t_tags \
                        in zip(arc_preds, lbl_preds, tag_preds, head_batch, label_batch, pos_tag_batch):
                    u_scorer(arcs=(p_arcs, t_arcs))
                    l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))
                    t_scorer(arcs=(p_tags, t_tags))
            else:
                for p_arcs, p_lbls, t_arcs, t_lbls in zip(arc_preds, lbl_preds, head_batch, label_batch):
                    u_scorer(arcs=(p_arcs, t_arcs))
                    l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))

            mean_loss(loss_value)
            out_str = tf_str.format(len(batch), mean_loss.score, u_scorer.score, l_scorer.score, t_scorer.score)
            pbar.set_description(out_str)
            pbar.update(len(batch))
    # if num_labels is None:
    #     conf_matrix = [[]]
    # else:
    #     conf_matrix = l_scorer.conf_matrix.tolist()
    # conf_matrix = [[]]
    stats = {label_stat('mean_loss'): mean_loss.score,
             label_stat('uas'): u_scorer.score,
             label_stat('las'): l_scorer.score,
             label_stat('pos'): t_scorer.score}
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
                        help='Whether to print additional info such '
                        'as model and vocabulary info.')
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
    
    t_data = dataset_to_cols(t_set, conf)

    # instantiate vocabs
    v_word = Vocab(out_size=conf.vocab.size, threshold=conf.vocab.threshold)
    v_arcs = UDepVocab()

    # fit vocabs to data
    if conf.subword:
        # if working on subwords, t_data.text is of depth 3: sents, words, chars
        # so we need to chain to pass a flat list of char ngrams
        v_word = v_word.fit(chain.from_iterable(chain.from_iterable(t_data.text)))
    else:
        v_word = v_word.fit(chain.from_iterable(t_data.text))

    # when also training the model as a part of speech tagger
    if conf.model.predict_pos:
        vocab_tup = namedtuple('Vocabs', ('text', 'arcs', 'pos_tags'))
        v_pos = UPOSVocab()
        vocabs = vocab_tup(v_word, v_arcs, v_pos)
    else:
        vocab_tup = namedtuple('Vocabs', ('text', 'arcs'))
        vocabs = vocab_tup(v_word, v_arcs)

    # visualise vocabs
    if conf.verbose:
        for v in vocabs:
            print(v)
            visualise_dict(v.index, num_items=50)

    train_rows = data_to_rows(t_data, vocabs, conf)

    v_data = dataset_to_cols(v_set, conf)
    dev_rows = data_to_rows(v_data, vocabs, conf)

    if conf.subword:
        conf.model.encoder.embedder.word_encoder.vocab_size = len(v_word)
    else:
        conf.model.encoder.embedder.in_sizes = [len(v_word)]
    conf.model.num_labels = len(v_arcs)
    if conf.model.predict_pos:
        conf.model.num_pos_tags = len(v_pos)

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
    stats_filename = '%s.stats' % filename
    dataset_folder = os.path.join(outfolder, conf.dataset.name.lower())
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    lang_folder = os.path.join(dataset_folder, conf.dataset.lang.lower())
    if not os.path.isdir(lang_folder):
        os.mkdir(lang_folder)
    blueprint_path = os.path.join(lang_folder, blueprint_filename)
    model_path = os.path.join(lang_folder, model_filename)
    vocab_path = os.path.join(lang_folder, vocab_filename)
    stats_path = os.path.join(lang_folder, stats_filename)

    # prepare for results
    stats = defaultdict(list)

    def on_epoch_end(epoch, epoch_stats, improved):
        if improved:
            print(' Saving model..')
            chainer.serializers.save_npz(model_path, built_conf.model)
        for key, value in epoch_stats.items():
            stats[key].append(value)
        with open(stats_path, 'w') as stats_out:
            stats_out.write(yaml.dump(dict(stats)))

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
            dill.dump(vocabs, pf)
        conf.vocab_path = vocab_path
        print('Writing blueprint to %s' % blueprint_path)
        conf.to_file(blueprint_path)
    except Exception:
        os.remove(model_path)
        os.remove(vocab_path)
        os.remove(blueprint_path)
