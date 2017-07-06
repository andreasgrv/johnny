# taken from https://github.com/LxMLS/lxmls-toolkit/blob/master/lxmls/parsing/dependency_decoder.py
# MIT LICENSED
import numpy as np


class DependencyDecoder:
    """
    Dependency decoder class
    """

    def __init__(self):
        self.verbose = False

    def parse_marginals_nonproj(self, scores):
        """
        Compute marginals and the log-partition function using the matrix-tree theorem
        """
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        nw = nr - 1

        s = np.matrix(scores)
        lap = np.matrix(np.zeros((nw+1, nw+1)))
        for m in range(1, nw+1):
            d = 0.0
            for h in range(0, nw+1):
                if m != h:
                    d += np.exp(s[h, m])
                    lap[h, m] = -np.exp(s[h, m])
            lap[m, m] = d
        r = lap[0, 1:]
        minor = lap[1:, 1:]

        # logZ = np.linalg.slogdet(minor)[1]
        logZ = np.log(np.linalg.det(minor))
        invmin = np.linalg.inv(minor)
        marginals = np.zeros((nw+1, nw+1))
        for m in range(1, nw+1):
            marginals[0, m] = np.exp(s[0, m]) * invmin[m-1, m-1]
            for h in range(1, nw+1):
                if m != h:
                    marginals[h, m] = np.exp(s[h, m]) * (invmin[m-1, m-1] - invmin[m-1, h-1])

        return marginals, logZ

    def parse_proj(self, scores):
        """
        Parse using Eisner's algorithm.
        """

        # ----------
        # Solution to Exercise 4.3.6
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        N = nr - 1  # Number of words (excluding root).

        # Initialize CKY table.
        complete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
        incomplete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
        complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).
        incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).

        incomplete[0, :, 0] -= np.inf

        # Loop from smaller items to larger items.
        for k in range(1, N+1):
            for s in range(N-k+1):
                t = s + k

                # First, create incomplete items.
                # left tree
                incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
                incomplete[s, t, 0] = np.max(incomplete_vals0)
                incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
                # right tree
                incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
                incomplete[s, t, 1] = np.max(incomplete_vals1)
                incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

                # Second, create complete items.
                # left tree
                complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
                complete[s, t, 0] = np.max(complete_vals0)
                complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
                # right tree
                complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
                complete[s, t, 1] = np.max(complete_vals1)
                complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

        value = complete[0][N][1]
        heads = -np.ones(N + 1, dtype=int)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

        value_proj = 0.0
        for m in range(1, N+1):
            h = heads[m]
            value_proj += scores[h, m]

        return heads

        # End of solution to Exercise 4.3.6
        # ----------

    def backtrack_eisner(self, incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
        """
        Backtracking step in Eisner's algorithm.
        - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
        an end position, and a direction flag (0 means left, 1 means right). This array contains
        the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
        - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
        an end position, and a direction flag (0 means left, 1 means right). This array contains
        the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
        - s is the current start of the span
        - t is the current end of the span
        - direction is 0 (left attachment) or 1 (right attachment)
        - complete is 1 if the current span is complete, and 0 otherwise
        - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
        head of each word.
        """
        if s == t:
            return
        if complete:
            r = complete_backtrack[s][t][direction]
            if direction == 0:
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
                return
            else:
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
                return
        else:
            r = incomplete_backtrack[s][t][direction]
            if direction == 0:
                heads[s] = t
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
                return
            else:
                heads[t] = s
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
                return

    def parse_nonproj(self, scores):
        """
        Parse using Chu-Liu-Edmonds algorithm.
        """
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        nw = nr - 1

        curr_nodes = np.ones(nw+1, int)
        reps = []
        old_I = -np.ones((nw+1, nw+1), int)
        old_O = -np.ones((nw+1, nw+1), int)
        for i in range(0, nw+1):
            reps.append({i: 0})
            for j in range(0, nw+1):
                old_I[i, j] = i
                old_O[i, j] = j
                if i == j or j == 0:
                    continue

        if self.verbose:
            print("Starting C-L-E...\n")

        scores_copy = scores.copy()
        final_edges = self.chu_liu_edmonds(scores_copy, curr_nodes, old_I, old_O, {}, reps)
        heads = np.zeros(nw+1, int)
        heads[0] = -1
        for key in final_edges.keys():
            ch = key
            pr = final_edges[key]
            heads[ch] = pr

        return heads

    def chu_liu_edmonds(self, scores, curr_nodes, old_I, old_O, final_edges, reps):
        """
        Chu-Liu-Edmonds algorithm
        """

        # need to construct for each node list of nodes they represent (here only!)
        nw = np.size(curr_nodes) - 1

        # create best graph
        par = -np.ones(nw+1, int)
        for m in range(1, nw+1):
            # only interested in current nodes
            if 0 == curr_nodes[m]:
                continue
            max_score = scores[0, m]
            par[m] = 0
            for h in range(nw+1):
                if m == h:
                    continue
                if 0 == curr_nodes[h]:
                    continue
                if scores[h, m] > max_score:
                    max_score = scores[h, m]
                    par[m] = h

        if self.verbose:
            print("After init\n")
            for m in range(0, nw+1):
                if 0 < curr_nodes[m]:
                    print("{0}|{1} ".format(par[m], m))
            print("\n")

        # find a cycle
        cycles = []
        added = np.zeros(nw+1, int)
        for m in range(0, nw+1):
            if np.size(cycles) > 0:
                break
            if added[m] or 0 == curr_nodes[m]:
                continue
            added[m] = 1
            cycle = {m: 0}
            l = m
            while True:
                if par[l] == -1:
                    added[l] = 1
                    break
                if par[l] in cycle:
                    cycle = {}
                    lorg = par[l]
                    cycle[lorg] = par[lorg]
                    added[lorg] = 1
                    l1 = par[lorg]
                    while l1 != lorg:
                        cycle[l1] = par[l1]
                        added[l1] = True
                        l1 = par[l1]
                    cycles.append(cycle)
                    break
                cycle[l] = 0
                l = par[l]
                if added[l] and (l not in cycle):
                    break
                added[l] = 1

        # get all edges and return them
        if np.size(cycles) == 0:
            for m in range(0, nw+1):
                if 0 == curr_nodes[m]:
                    continue
                if par[m] != -1:
                    pr = old_I[par[m], m]
                    ch = old_O[par[m], m]
                    final_edges[ch] = pr
                else:
                    final_edges[0] = -1
            return final_edges

        max_cyc = 0
        wh_cyc = 0
        for cycle in cycles:
            if np.size(cycle.keys()) > max_cyc:
                max_cyc = np.size(cycle.keys())
                wh_cyc = cycle

        cycle = wh_cyc
        cyc_nodes = list(cycle.keys())
        rep = next(iter(cyc_nodes))

        if self.verbose:
            print("Found Cycle\n")
            for node in cyc_nodes:
                print("{0} ".format(node))
            print("\n")

        cyc_weight = 0.0
        for node in cyc_nodes:
            cyc_weight += scores[par[node], node]

        for i in range(0, nw+1):
            if 0 == curr_nodes[i] or (i in cycle):
                continue

            max1 = -np.inf
            wh1 = -1
            max2 = -np.inf
            wh2 = -1

            for j1 in cyc_nodes:
                if scores[j1, i] > max1:
                    max1 = scores[j1, i]
                    wh1 = j1

                # cycle weight + new edge - removal of old
                scr = cyc_weight + scores[i, j1] - scores[par[j1], j1]
                if scr > max2:
                    max2 = scr
                    wh2 = j1

            scores[rep, i] = max1
            old_I[rep, i] = old_I[wh1, i]
            old_O[rep, i] = old_O[wh1, i]
            scores[i, rep] = max2
            old_O[i, rep] = old_O[i, wh2]
            old_I[i, rep] = old_I[i, wh2]

        rep_cons = []
        for i in range(0, np.size(cyc_nodes)):
            rep_con = {}
            keys = sorted(list(reps[int(cyc_nodes[i])].keys()))
            if self.verbose:
                print("{0}: ".format(cyc_nodes[i]))
            for key in keys:
                rep_con[key] = 0
                if self.verbose:
                    print("{0} ".format(key))
            rep_cons.append(rep_con)
            if self.verbose:
                print("\n")

        # don't consider not representative nodes
        # these nodes have been folded
        for node in cyc_nodes[1:]:
            curr_nodes[node] = 0
            for key in reps[int(node)]:
                reps[int(rep)][key] = 0

        self.chu_liu_edmonds(scores, curr_nodes, old_I, old_O, final_edges, reps)

        # check each node in cycle, if one of its representatives
        # is a key in the final_edges, it is the one.
        if self.verbose:
            print(final_edges)
        wh = -1
        found = False
        for i in range(0, np.size(rep_cons)):
            if found:
                break
            for key in rep_cons[i]:
                if found:
                    break
                if key in final_edges:
                    wh = cyc_nodes[i]
                    found = True
        l = par[wh]
        while l != wh:
            ch = old_O[par[l]][l]
            pr = old_I[par[l]][l]
            final_edges[ch] = pr
            l = par[l]

        return final_edges

# code modified from chainer https://github.com/chainer/chainer/blob/master/chainer/links/connection/n_step_lstm.py
import numpy
import six

from chainer import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_lstm as rnn
from chainer.initializers import normal
from chainer import link
from chainer.links.connection import n_step_rnn
from chainer.utils import argument
from chainer import variable


class NStepLSTMBase(link.ChainList):
    """Base link class for Stacked LSTM/BiLSTM links.
    This link is base link class for :func:`chainer.links.NStepLSTM` and
    :func:`chainer.links.NStepBiLSTM`.
    This link's behavior depends on argument, ``use_bi_direction``.
    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        use_bi_direction (bool): if ``True``, use Bi-directional LSTM.
    .. seealso::
        :func:`chainer.functions.n_step_lstm`
        :func:`chainer.functions.n_step_bilstm`
    """

    def __init__(self, n_layers, in_size, out_size, dropout, use_bi_direction,
                 **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, use_cudnn='use_cudnn argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        weights = []
        direction = 2 if use_bi_direction else 1
        for i in six.moves.range(n_layers):
            for di in six.moves.range(direction):
                weight = link.Link()
                with weight.init_scope():
                    for j in six.moves.range(8):
                        if i == 0 and j < 4:
                            w_in = in_size
                        elif i > 0 and j < 4:
                            w_in = out_size * direction
                        else:
                            w_in = out_size
                        w = variable.Parameter(
                            normal.Normal(numpy.sqrt(1. / w_in)),
                            (out_size, w_in))
                        b = variable.Parameter(0, (out_size,))
                        setattr(weight, 'w%d' % j, w)
                        setattr(weight, 'b%d' % j, b)
                weights.append(weight)

        super(NStepLSTMBase, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.out_size = out_size
        self.direction = direction
        self.rnn = rnn.n_step_bilstm if use_bi_direction else rnn.n_step_lstm

    def init_hx(self, xs):
        first = xs[0]
        shape = (self.n_layers * self.direction, first.shape[0], self.out_size)
        with cuda.get_device_from_id(self._device_id):
            hx = variable.Variable(self.xp.zeros(shape, dtype=first.dtype))
        return hx

    def __call__(self, hx, cx, xs, **kwargs):
        """__call__(self, hx, cx, xs)
        Calculate all hidden states and cell states.
        .. warning::
           ``train`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            hx (~chainer.Variable or None): Initial hidden states. If ``None``
                is specified zero-vector is used.
            cx (~chainer.Variable or None): Initial cell states. If ``None``
                is specified zero-vector is used.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        """
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        assert isinstance(xs, (list, tuple))
        # indices = n_step_rnn.argsort_list_descent(xs)

        # xs = n_step_rnn.permutate_list(xs, indices, inv=False)
        if hx is None:
            hx = self.init_hx(xs)
        # else:
        #     hx = permutate.permutate(hx, indices, axis=1, inv=False)

        if cx is None:
            cx = self.init_hx(xs)
        # else:
        #     cx = permutate.permutate(cx, indices, axis=1, inv=False)

        # trans_x = transpose_sequence.transpose_sequence(xs)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self]

        hy, cy, trans_y = self.rnn(
            self.n_layers, self.dropout, hx, cx, ws, bs, xs)

        # hy = permutate.permutate(hy, indices, axis=1, inv=True)
        # cy = permutate.permutate(cy, indices, axis=1, inv=True)
        # ys = transpose_sequence.transpose_sequence(trans_y)
        # ys = n_step_rnn.permutate_list(ys, indices, inv=True)

        return hy, cy, trans_y
