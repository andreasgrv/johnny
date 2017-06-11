import numpy as np


class BucketManager(object):

    DATA_KEY = 'data'
    INDEX_KEY = 'index'
    END_INDEX_KEY = 'end_index'


    def __init__(self, data, bucket_width, max_len, min_len=1, batch_size=64,
                 shuffle=True, row_key=None):
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
        # by default we use the length of the first entry in the row to sort by
        self.row_key = row_key or (lambda x: len(x))
        self.batch_count = 0

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
            return data
        else:
            raise StopIteration

    def shuffle_bucket_contents(self):
        """Shuffles entries inside each bucket"""
        for buck_indx in range(self.num_buckets):
            target = self.buckets[buck_indx]
            np.random.shuffle(target[self.DATA_KEY])

    def reset(self, shuffle=None):
        """Resets status"""
        shuffle = shuffle or self.shuffle
        if shuffle:
            self.shuffle_bucket_contents()
        for buck_indx in range(self.num_buckets):
            target = self.buckets[buck_indx]
            # rewinds index of each bucket
            target[self.INDEX_KEY] = 0
            self.left_samples[buck_indx] = target[self.END_INDEX_KEY]
        self.batch_count = 0

    def sample(self):
        more_to_go = self.total_left
        if more_to_go:
            probs = self.left_samples/more_to_go
            which_bucket = np.random.choice(self.num_buckets, 1, p=probs)[0]
            #print('which_bucket',which_bucket)
            bucket = self.buckets[which_bucket]
            index = bucket[self.INDEX_KEY]
            left_over = bucket[self.END_INDEX_KEY] - index
            num_samples = min((self.batch_size, left_over))
            data = bucket[self.DATA_KEY][index : index + num_samples]
            bucket[self.INDEX_KEY] += num_samples
            self.left_samples[which_bucket] -= num_samples
            return data
        return None

    @property
    def total_left(self):
        return int(sum(self.left_samples))

shades = [' ', '░', '▒', '▓', '█']

# def shade(words, probs):
#     return ' '.join([shades[int(prob/0.2)]*len(word)
#                      for word, prob in zip(words, probs)])
def shade(probs):
    return ''.join([shades[int(prob/0.201)] for prob in probs])

bars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇']

def bar(probs):
    return ''.join([bars[int(prob/0.151)] for prob in probs])

def discrete_print(string):
    newlines = string.count('\n') + 1
    filler = '\n' * newlines
    cursor_move = '\033[%dA\033[100D' % newlines
    return '%s%s\n%s%s' % (filler, cursor_move, string, cursor_move)
