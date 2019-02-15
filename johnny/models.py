import six
import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer
from time import sleep
from itertools import chain
from chainer import Variable, cuda
from johnny.misc import bar, discrete_print
from johnny.extern import DependencyDecoder
from johnny.vocab import UDepVocab


# TODO Check multiple roots issue
# TODO Check constraint algorithms + optimise
# TODO Maybe switch to using predicted arcs towards end of training
# TODO Think of ways of avoiding self prediction
class GraphParser(chainer.Chain):

    MIN_PAD = -1e4
    TREE_OPTS = ['none', 'chu', 'eisner']

    def __init__(self,
                 encoder,
                 num_labels=46,
                 num_pos_tags=20,
                 mlp_arc_units=100,
                 mlp_lbl_units=100,
                 mlp_tag_units=100,
                 arc_dropout=0.0,
                 lbl_dropout=0.5,
                 tag_dropout=0.5,
                 predict_pos=False,
                 treeify='chu',
                 visualise=False,
                 debug=False
                 ):

        super(GraphParser, self).__init__()
        self.num_labels = num_labels
        self.num_pos_tags = num_pos_tags
        self.mlp_arc_units = mlp_arc_units
        self.mlp_lbl_units = mlp_lbl_units
        self.mlp_tag_units = mlp_tag_units
        self.arc_dropout = arc_dropout
        self.lbl_dropout = lbl_dropout
        self.tag_dropout = tag_dropout
        self.predict_pos = predict_pos
        self.treeify = treeify.lower()
        self.visualise = visualise
        self.debug = debug
        self.sleep_time = 0.

        assert(treeify in self.TREE_OPTS)
        self.unit_mult = 2 if encoder.use_bilstm else 1

        with self.init_scope():
            self.encoder = encoder

            # head prediction
            embedding_units = self.unit_mult * self.encoder.num_units
            # head
            self.H_arc = L.Linear(embedding_units, self.mlp_arc_units)
            # dependent
            self.D_arc = L.Linear(embedding_units, self.mlp_arc_units)
            # output
            self.vT = L.Linear(self.mlp_arc_units, 1)

            # label prediction
            self.W_lbl = L.Linear(self.mlp_arc_units, self.mlp_lbl_units)
            # output
            self.V_lblT = L.Linear(self.mlp_lbl_units, self.num_labels)

            # pos tag prediction
            if self.predict_pos:
                self.W_tag = L.Linear(embedding_units, self.mlp_tag_units)
                # output
                self.V_tagT = L.Linear(self.mlp_lbl_units, self.num_labels)

    def _predict_pos_tags(self, sent_embeds, batch_stats, pos_tags=None):
        """ Predict the part of speech tag for each word in the sentence."""
        batch_size, seq_lengths, max_sent_len = batch_stats

        calc_loss = pos_tags is not None
        if calc_loss:
            pos_tags = chainer.Variable(
                        self.xp.array(tuple(chain.from_iterable(pos_tags)))
                       )

        w_tag = self.W_tag(sent_embeds)

        t_act = F.tanh(w_tag)

        if self.tag_dropout > 0.:
            t_act = F.dropout(t_act, ratio=self.tag_dropout)

        tag_logits = self.V_tagT(t_act)

        if calc_loss:
            tags_loss = F.sum(F.softmax_cross_entropy(tag_logits, pos_tags, reduce='no'))
            self.loss += tags_loss

        return tag_logits

    def _predict_heads(self, sent_embeds, root_embeds, batch_stats, heads=None):
        """ For each token in the sentence predict which token in the sentence
        is its head."""

        batch_size, seq_lengths, max_sent_len = batch_stats

        calc_loss = heads is not None

        if calc_loss:
            #TODO: Wrap heads in variable
            heads = chainer.Variable(
                        self.xp.array(tuple(chain.from_iterable(heads)))
                    )
        
        # In order to predict which head is most probable for a given word
        # For each token in the sentence we get a vector represention
        # for that token as a head, and another for that token as a dependent.
        # The idea here is that we only need to calculate these once
        # and then reuse them to get all combinations
        # ------------------------------------------------------
        # In g(a_j, a_i) we note that we can precompute the matrix multiplications
        # for each word, we consider all possible heads
        # we can pre-calculate U * a_j , for all a_j
        # head activations for each token lstm activation
        # h_arc -> head representation | d_arc -> dependent representation
        h_arc = self.H_arc(sent_embeds)
        h_arc_root = self.H_arc(root_embeds)

        d_arc = self.D_arc(sent_embeds)

        def to_batched_heads(heads, roots, seq_lens):
            """Put each root embedding before its batch."""
            batched = F.vstack([roots, heads])

            permute_roots = np.arange(roots.shape[0])
            permute_roots[1:] += np.cumsum(seq_lens[:-1])

            permute_heads = np.arange(heads.shape[0]) + 1
            permute_heads += np.repeat(np.arange(len(seq_lens)), seq_lens)

            permute_indices = np.hstack([permute_roots, permute_heads])

            return F.permutate(batched, permute_indices, inv=True)

        def repeat_deps(x, seq_lens):
            # Extend the deps representations
            seq = tuple(np.repeat(seq_lens + 1, seq_lens).tolist())
            return F.repeat(x, seq, axis=0)

        def tile_heads(x, seq_lens):
            splits = np.cumsum(seq_lens[:-1] + 1)
            return F.vstack(F.tile(part, (part.shape[0] - 1, 1))
                            for part in F.split_axis(x, splits, axis=0))

        h_embs = to_batched_heads(h_arc, h_arc_root, seq_lengths)
        h_embs = tile_heads(h_embs, seq_lengths)

        d_embs = repeat_deps(d_arc, seq_lengths)

        arc_logit = F.tanh(d_embs + h_embs)

        if self.arc_dropout > 0.:
            arc_logit = F.dropout(arc_logit, ratio=self.arc_dropout)

        self.label_logit = arc_logit

        arc_logit = self.vT(arc_logit)

        split_to_sents = np.repeat(seq_lengths + 1, seq_lengths)
        split_to_sents = np.cumsum(split_to_sents[:-1])
        self.split_to_sents = split_to_sents

        decision_logits = F.split_axis(arc_logit, split_to_sents, axis=0)

        arc_logits = F.pad_sequence(decision_logits, max_sent_len + 1, padding=self.MIN_PAD)
        # Get rid of 3rd dimension
        arc_logits = F.squeeze(arc_logits, axis=2)

        if calc_loss:
            head_loss = F.sum(F.softmax_cross_entropy(arc_logits, heads, reduce='no'))
            self.loss += head_loss

        return arc_logits

    def _predict_labels(self, sent_embeds, pred_heads, gold_heads, batch_stats,
                        labels=None):
        """ Predict the label for each of the arcs predicted in _predict_heads."""
        batch_size, seq_lengths, max_sent_len = batch_stats

        calc_loss = labels is not None
        if calc_loss:
            labels = chainer.Variable(
                         self.xp.array(tuple(chain.from_iterable(labels)))
                     )

        offsets = self.xp.pad(self.split_to_sents, (1, 0), 'constant', constant_values=0)

        if gold_heads is None:
            # Use predictions at test time
            arc_indices = offsets + pred_heads
        else:
            # Use gold heads at train time (teacher forcing)
            arc_indices = offsets + self.xp.array(tuple(chain.from_iterable(gold_heads)))

        label_logits = self.label_logit[arc_indices]
        # Choose representations that contain head and label
        # label_logit is packed 1D from (batch_size, variable seq len, hidden size)
        # we know variable length from seq_lengths
        w_lbl = self.W_lbl(label_logits)
        w_lbl = F.tanh(w_lbl)

        if self.lbl_dropout > 0.:
            w_lbl = F.dropout(w_lbl, ratio=self.lbl_dropout)

        lbl_logits = self.V_lblT(w_lbl)

        if calc_loss:
            label_loss = F.sum(F.softmax_cross_entropy(lbl_logits, labels, reduce='no'))
            self.loss += label_loss

        return lbl_logits

    def __call__(self, *inputs, **kwargs):
        """ Expects a batch of sentences 
        so a list of K sentences where each sentence
        is a 2-tuple of indices of words, indices of pos tags.
        Example of 2 sentences:
            [([1,5,2], [4,7,1]), ([1,2], [3,4])]
              w1 w2      p1 p2
              |------- s1 -------|

        w = word, p = pos tag, s = sentence

        This is as slow as the longest sentence - so bucketing sentences
        of same size can speed up training/inference.
        """
        assert(len(inputs) >= 1)
        if kwargs:
            lens = [len(inp) for inp in kwargs.values() if inp]
            assert(all(l == lens[0] for l in lens))

        heads = kwargs.get('heads', None)
        labels = kwargs.get('labels', None)
        pos_tags = kwargs.get('pos_tags', None)


        sentence_embeddings = self.encoder(*inputs)

        self.loss = 0.

        batch_stats = (self.encoder.batch_size,
                       self.encoder.sequence_lengths,
                       self.encoder.max_seq_len)

        arcs = self._predict_heads(sentence_embeddings,
                                   self.encoder.root_embeddings,
                                   batch_stats,
                                   heads=heads)

        if self.debug or self.visualise:
            self.arcs = cuda.to_cpu(F.softmax(arcs).data)

        # treeification is carried out before prediction labels
        # just in case we decide we want to use treeified predictions
        # of heads as input to the label predictor
        if self.treeify != 'none':
            # TODO: check multiple roots issue
            # We process the head scores to apply tree constraints
            arcs = cuda.to_cpu(arcs.data)
            # arcs are batch_size x sent_len + 1 x sent_len
            # axis 1 has the scores over the sentence
            # axis 2 is one shorter because we don't predict for root
            dd = DependencyDecoder()
            # sent length not taking root into account
            sent_lengths = self.encoder.sequence_lengths
            split_idxs = np.cumsum(sent_lengths[:-1])

            arc_preds = []
            for l, score_mat in zip(sent_lengths, np.split(arcs, split_idxs, axis=0)):
                # remove fallout from batch size
                trunc_score_mat = score_mat[:, :l+1]
                # DependencyDecoder expects a square matrix - fill root col with zeros
                trunc_score_mat = np.pad(trunc_score_mat, ((1, 0), (0, 0)), 'constant').T
                if self.treeify == 'chu':
                    # Just remove cycles, non-projective trees are ok
                    nproj_arcs = dd.parse_nonproj(trunc_score_mat)[1:]
                    arc_preds.append(nproj_arcs)

                # arc_preds = np.array([dd.parse_nonproj(each)[1:] for each in pd_arcs])
                elif self.treeify == 'eisner':
                    # Remove cycles and make sure trees are projective
                    proj_arcs = dd.parse_proj(trunc_score_mat)[1:]
                    arc_preds.append(proj_arcs)
                else:
                    raise ValueError('Unexpected method')
            arc_preds = np.hstack(arc_preds)
        else:
            # We ignore tree constraints - head predictions may create cycles
            # we pass predict_labels the gpu object
            arcs = arcs.data
            arc_preds = self.xp.argmax(arcs, axis=1)
            arc_preds = cuda.to_cpu(arc_preds)

        lbls = self._predict_labels(sentence_embeddings,
                                    arc_preds,
                                    heads,
                                    batch_stats,
                                    labels=labels)

        if self.predict_pos:
            pos_tags = self._predict_pos_tags(sentence_embeddings,
                                              batch_stats,
                                              pos_tags=pos_tags)

        if self.debug or self.visualise:
            self.lbls = cuda.to_cpu(F.softmax(lbls).data)

        lbls = cuda.to_cpu(lbls.data)

        # we only bother actually getting the softmax values
        # if we are to visualise the results
        if self.visualise:
            self._visualise(self.arcs[0], self.lbls[0], heads[0], labels[0])

        # normalize loss over all tokens seen
        total_tokens = np.sum(self.encoder.sequence_lengths)
        self.loss = self.loss / total_tokens

        lbl_preds = np.argmax(lbls, axis=1)

        split_idxs = np.cumsum(self.encoder.sequence_lengths[:-1])

        arc_preds = [arc_p.tolist()
                     for arc_p
                     in np.split(arc_preds, split_idxs)]
        lbl_preds = [lbl_p.tolist()
                     for lbl_p
                     in np.split(lbl_preds, split_idxs)]

        if self.predict_pos:
            tags = cuda.to_cpu(pos_tags.data)
            tag_preds = np.argmax(tags, axis=1)
            tag_preds = [tag_p.tolist()
                         for tag_p
                         in np.split(tag_preds, split_idxs)]

            return arc_preds, lbl_preds, tag_preds

        return arc_preds, lbl_preds

    def _visualise(self, arcs, lbls, gold_heads, gold_labels):
        max_sent_len = len(arcs[1])
        one_hot_index = self.xp.zeros(max_sent_len, dtype=self.xp.float32)
        one_hot_arc = self.xp.zeros(max_sent_len+1, dtype=self.xp.float32)
        one_hot_lbl = self.xp.zeros(self.num_labels, dtype=self.xp.float32)
        for i in range(max_sent_len):
            correct_head_index = gold_heads[i]
            one_hot_arc[correct_head_index] = 1.
            one_hot_index[i] = 1.
            correct_lbl_index = gold_labels[i]
            one_hot_lbl[correct_lbl_index] = 1.
            six.print_(discrete_print('\n\nCur index : %-110s\nReal head : %-110s\nPred head : %-110s\n\n'
                                 'Real label: %-110s\nPred label: %-110s\n\n'
                                 'Sleep time: %.2f - change with up and down arrow keys') % (
                 '[%s] %d' % (bar(one_hot_index[:90]), i),
                 '[%s] %d |%d|' % (bar(one_hot_arc[:90]), correct_head_index, abs(correct_head_index - i)),
                    '[%s]' % bar(arcs[:, i].reshape(-1)[:90]),
                    '[%s] %s' % (bar(one_hot_lbl), UDepVocab.TAGS[correct_lbl_index]),
                    '[%s]' % bar(lbls[:, i].reshape(-1)),
                    self.sleep_time),
                  end='', flush=True)
            # reset values
            one_hot_index[i] = 0.
            one_hot_arc[correct_head_index] = 0.
            one_hot_lbl[correct_lbl_index] = 0.
            sleep(self.sleep_time)
