from __future__ import division
import numpy as np


class Average(object):

    def __init__(self, label=None):
        self.count = 0
        self.cumsum = 0.
        self.label = label or 'Average'

    def __call__(self, value):
        self.count += 1
        self.cumsum += value
        return self.cumsum / self.count if self.count else 0.0

    def reset(self):
        self.count = 0
        self.cumsum = 0.

    @property
    def score(self):
        return self.cumsum / self.count if self.count else 0.0


class UAS(Average):

    '''Unlabelled Attachment Score - Scorer'''

    def __init__(self, label='UAS'):
        Average.__init__(self, label)

    def __call__(self, arcs=None):
        pred_arcs, true_arcs = arcs
        preds = np.asarray(pred_arcs)
        truth = np.asarray(true_arcs)
        correct = float(np.sum(preds == truth))
        self.count += len(preds)
        self.cumsum += correct
        return self.cumsum / self.count if self.count else 0.0


class LAS(Average):

    '''Labelled Attachment Score - Scorer'''

    def __init__(self, label='LAS', num_labels=None):
        Average.__init__(self, label)
        if num_labels is not None:
            self.conf_matrix = np.zeros((num_labels, num_labels),
                                        dtype=np.int32)

    def __call__(self, arcs=None, labels=None):
        pred_arcs, true_arcs = arcs
        pred_labels, true_labels = labels
        p_arcs = np.asarray(pred_arcs)
        t_arcs = np.asarray(true_arcs)
        p_labels = np.asarray(pred_labels)
        t_labels = np.asarray(true_labels)
        correct = float(np.sum((p_arcs == t_arcs) & (p_labels == t_labels)))
        self.count += len(p_arcs)
        self.cumsum += correct
        return self.cumsum / self.count if self.count else 0.0
