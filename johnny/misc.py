# -*- coding: utf-8 -*-
import os
import six
import numpy as np
import yaml
import datetime
from johnny import EXP_ENV_VAR


class BucketManager(six.Iterator):

    DATA_KEY = 'data'
    INDEX_KEY = 'index'
    END_INDEX_KEY = 'end_index'


    def __init__(self, data, bucket_width, max_len, min_len=1, batch_size=64,
                 shuffle=True, right_leak=None, row_key=None, loop_forever=False):
        """
        data: a list of rows

        bucket_width: int - how much of a difference in length is tolerable - 
        hashed to the same bucket.

        max_len: the maximum length of a row or row entry - includes the
        maximum.

        min_len: the minumum length of a row or row entry - default value of 1
        means we start considering things of length 1 - if we have something
        empty we will get an index error because we aren't considering 0
        length objects.

        shuffle: whether to shuffle entries in each bucket or not

        row_key: callable - a function run on the row to compute a value to
        use to map it to a bucket
        """
        super(BucketManager, self).__init__()
        self.bucket_width = bucket_width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.right_leak = right_leak
        # by default we use the length of the first entry in the row to sort by
        self.row_key = row_key or (lambda x: len(x))
        self.loop_forever = loop_forever
        self.batch_count = 0
        self.seq_count = 0

        self.max_len = max_len
        self.min_len = min_len
        assert(self.bucket_width > 0)
        assert(self.max_len > 0)
        assert(self.min_len >= 0)
        # +1 because we want to also be able to store sequences of max_len
        # the range is [min_len, max_len]
        bucket_range = self.max_len - self.min_len + 1
        exact_fit = bucket_range // self.bucket_width 
        has_remainder = int(bucket_range % self.bucket_width > 0)
        self.num_buckets = exact_fit + has_remainder

        self.buckets = [{self.DATA_KEY: [], self.END_INDEX_KEY: 0, self.INDEX_KEY: 0}
                        for i in range(self.num_buckets)]

        self.left_samples = np.zeros(self.num_buckets, dtype=np.float32)

        for row in data:
            length = self.row_key(row)
            adjust_length = length - self.min_len
            which_bucket = (adjust_length // self.bucket_width)
            if which_bucket < 0:
                raise IndexError
            target = self.buckets[which_bucket]
            target[self.DATA_KEY].append(row)
            target[self.END_INDEX_KEY] += 1
            self.left_samples[which_bucket] += 1

        if self.shuffle:
            self.shuffle_bucket_contents()

    def __iter__(self):
        return self

    def __next__(self):
        data = self.sample()
        if data:
            self.batch_count += 1
            self.seq_count += len(data)
            return data
        else:
            if self.loop_forever:
                self.reset(self.shuffle)
                return self.__next__()
            else:
                raise StopIteration

    def shuffle_bucket_contents(self):
        """Shuffles entries inside each bucket"""
        for buck_indx in range(self.num_buckets):
            target = self.buckets[buck_indx]
            np.random.shuffle(target[self.DATA_KEY])

    def reset(self, shuffle=None):
        """Resets status"""
        shuffle = self.shuffle if shuffle is None else shuffle
        if shuffle:
            self.shuffle_bucket_contents()
        for buck_indx in range(self.num_buckets):
            target = self.buckets[buck_indx]
            # rewinds index of each bucket
            target[self.INDEX_KEY] = 0
            self.left_samples[buck_indx] = target[self.END_INDEX_KEY]

    def sample(self):
        more_to_go = self.total_left
        if more_to_go:
            probs = self.left_samples/more_to_go
            which_bucket = np.random.choice(self.num_buckets, 1, p=probs)[0]
            bucket = self.buckets[which_bucket]
            index = bucket[self.INDEX_KEY]
            left_over = bucket[self.END_INDEX_KEY] - index
            num_samples = min((self.batch_size, left_over))
            data = bucket[self.DATA_KEY][index : index + num_samples]
            bucket[self.INDEX_KEY] += num_samples
            self.left_samples[which_bucket] -= num_samples

            if self.right_leak:
                # if we are sampling from a nearly empty bucket
                # and there are others with more to go, combine batches
                # from different buckets
                leak_max_index = min(self.num_buckets - 1, which_bucket + self.right_leak)
                leak_extension = self.left_samples[which_bucket: leak_max_index]
                more_to_go = np.sum(leak_extension)
                leak_index = which_bucket
                cur_size = len(data)
                while(cur_size < self.batch_size and leak_index <= leak_max_index and more_to_go):
                    data = data[:]
                    bucket = self.buckets[leak_index]
                    index = bucket[self.INDEX_KEY]
                    left_over = bucket[self.END_INDEX_KEY] - index
                    num_samples = min((self.batch_size - cur_size, left_over))
                    more_data = bucket[self.DATA_KEY][index : index + num_samples]
                    bucket[self.INDEX_KEY] += num_samples
                    self.left_samples[leak_index] -= num_samples
                    data.extend(more_data)
                    more_to_go = np.sum(leak_extension)
                    cur_size = len(data)
                    leak_index += 1
            return data
        return None

    @property
    def total_left(self):
        return int(sum(self.left_samples))


class Experiment(object):

    MODEL_SUFFIX = '.model'
    VOCAB_SUFFIX = '.vocab'
    DATE_FORMAT = '%d-%m-%Y %H:%M:%S'

    def __init__(self, name, lang, model, **kwargs):
        self.name = name
        self.lang = lang
        self.model = model
        self.results = dict(train_timeline=[])
        self.timestamp = datetime.datetime.strftime(datetime.datetime.now(),
                                                    self.DATE_FORMAT)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return self.to_yaml()

    def add_train_entry(self, **kwargs):
        self.results['train_timeline'].append(dict(**kwargs))

    def set_test_results(self, results):
        self.results['test_results'] = results

    def to_yaml(self):
        return yaml.dump(self.__dict__, default_flow_style=False)

    @classmethod
    def from_yaml(cl, yaml_string):
        return cl(**yaml.load(yaml_string))

    @staticmethod
    def _get_dir_path(exp_folder):
        try:
            directory = exp_folder or os.environ[EXP_ENV_VAR]
        except KeyError:
            raise AttributeError('Either set exp_folder_path attribute '
            'to the folder the experiments should be saved to or set the '
            '%s environment variable' % EXP_ENV_VAR)
        if not os.path.isdir(directory):
            raise ValueError('%s directory does not exist' % directory)
        return directory

    @staticmethod
    def _create_exp_folder(dir_path):
        try:
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
        except Exception:
            raise ValueError('Could not create folder %s please check you '
                             'have specified the correct folder to save the '
                             'experiments in and that you have '
                             'write access' % dir_path)

    def save(self, exp_folder_path=None, verbose=False):
        directory = self._get_dir_path(exp_folder_path)
        dir_path = os.path.join(directory, self.lang)
        self._create_exp_folder(dir_path)
        filename = ('%s_%s' % (self.name, self.timestamp))
        self.filepath = os.path.join(dir_path, filename)

        if verbose:
            print('Writing experiment to %s' % self.filepath)
        with open(self.filepath, 'w') as f:
            f.write(self.to_yaml())

    @classmethod
    def load(cl, filename):
        with open(filename, 'r') as f:
            yml = yaml.load(f.read())
        return cl(**yml)

    @property
    def model_path(self):
        return '%s%s' % (self.filepath, self.MODEL_SUFFIX)

    @property
    def vocab_path(self):
        return '%s%s' % (self.filepath, self.VOCAB_SUFFIX)

        
SHADES = ' ░▒▓█'
BARS = ' ▁▂▃▄▅▆▇'

def shade(probs):
    return ''.join([SHADES[int(prob/0.201)] for prob in probs])

def bar(probs):
    return ''.join([BARS[int(prob/0.151)] for prob in probs])

def discrete_print(string):
    newlines = string.count('\n') + 1
    filler = '\n' * newlines
    cursor_move = '\033[%dA\033[100D' % newlines
    return '%s%s\n%s%s' % (filler, cursor_move, string, cursor_move)


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
