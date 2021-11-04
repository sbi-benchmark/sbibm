"""
Code by Josip Djolonga:
- https://github.com/josipd/torch-two-sample

Based on commit d0771287fa1ba820ad975f1f038bfd8e155d2b91
Contains minor modifications, e.g., to account for changes to PyTorch


BSD 3-Clause License

Copyright (c) 2017, Josip Djolonga
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the ETH Zurich nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Josip Djolonga  BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from torch.autograd import Function, Variable
from torch.nn.functional import relu, softmax


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.0:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2)
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1.0 / norm)


NINF = -1e5  # TODO(josipd): Implement computation with negative infinities.


def logsumexp(x, dim):
    """Compute the log-sum-exp in a numerically stable way.
    Arguments
    ---------
    x : :class:`torch:torch.Tensor`
    dim : int
        The dimension along wich the operation should be computed.
    Returns
    --------
    :class:`torch:torch.Tensor`
        The dimension along which the sum is done is not squeezed.
    """
    x_max = torch.max(x, dim, keepdim=True)[0]
    return (
        torch.log(torch.sum(torch.exp(x - x_max.expand_as(x)), dim, keepdim=True))
        + x_max
    )


def logaddexp(x, y):
    """Compute log(e^x + e^y) element-wise in a numerically stable way.
    The arguments have to be of equal dimension.
    Arguments
    ---------
    x : :class:`torch:torch.Tensor`
    y : :class:`torch:torch.Tensor`"""
    maxes = torch.max(x, y)
    return torch.log(torch.exp(x - maxes) + torch.exp(y - maxes)) + maxes


def compute_bwd(node_pot, msg_in):
    """Compute the new backward message from the given node potential and
    incoming message."""
    node_pot = node_pot.unsqueeze(1)
    msg_out = msg_in.clone()
    msg_out[:, 1:] = logaddexp(
        msg_out[:, 1:], node_pot.expand_as(msg_in[:, :-1]) + msg_in[:, :-1]
    )
    # Normalize for numerical stability.
    return msg_out - logsumexp(msg_out, 1).expand_as(msg_out)


def compute_fwd(node_pot, msg_in):
    """Compute the new forward message from the given node potential and
    incoming message."""
    node_pot = node_pot.unsqueeze(1)
    msg_out = msg_in.clone()
    msg_out[:, :-1] = logaddexp(
        msg_out[:, :-1], node_pot.expand_as(msg_in[:, 1:]) + msg_in[:, 1:]
    )
    # Normalize for numerical stability.
    return msg_out - logsumexp(msg_out, 1).expand_as(msg_out)


def inference_cardinality(node_potentials, cardinality_potential):
    r"""Perform inference in a graphical model of the form
    .. math::
        p(x) \propto \exp( \sum_{i=1}^n x_iq_i + f(\sum_{i=1}^n x_i) ),
    where :math:`x` is a binary random variable. The vector :math:`q` holds the
    node potentials, while :math:`f` is the so-called cardinality potential.
    Arguments
    ---------
    node_potentials: :class:`torch:torch.autograd.Variable`
        The matrix holding the per-node potentials :math:`q` of size
        ``(batch_size, n)``.
    cardinality_potentials: :class:`torch:torch.autograd.Variable`
        The cardinality potential.
        Should be of size ``(batch_size, n_potentials)``.
        In each row, column ``i`` holds the value :math:`f(i)`.
        If it happens ``n_potentials < n + 1``, the remaining positions are
        assumed to be equal to ``-inf`` (i.e., are given zero probability)."""

    def create_var(val, *dims):
        """Helper to initialize a variable on the right device."""
        if node_potentials.is_cuda:
            tensor = torch.cuda.FloatTensor(*dims)
        else:
            tensor = torch.FloatTensor(*dims)
        tensor.fill_(val)
        return Variable(tensor, requires_grad=False)

    batch_size, dim_node = node_potentials.size()
    assert batch_size == cardinality_potential.size()[0]

    fmsgs = []
    fmsgs.append(cardinality_potential.clone())
    for i in range(dim_node - 1):
        fmsgs.append(compute_fwd(node_potentials[:, i], fmsgs[-1]))
    fmsgs.append(create_var(NINF, cardinality_potential.size()))

    bmsgs = []
    bmsgs.append(create_var(NINF, cardinality_potential.size()))
    bmsgs[0][:, 0] = 0
    bmsgs[0][:, 1] = node_potentials[:, dim_node - 1]
    for i in reversed(range(1, dim_node)):
        bmsgs.insert(0, compute_bwd(node_potentials[:, i - 1], bmsgs[0]))
    bmsgs.insert(0, create_var(NINF, cardinality_potential.size()))

    # Construct pairwise beliefs (without explicitly instantiating the D^2
    # size matrices), then sum the diagonal to get b0, and the off-diagonal
    # to get b1.  b1/(b0+b1) gives marginal for original y_d for all except
    # the last variable, y_D.  we need to special case it, because there is
    # no pairwise potential that represents \theta_D -- it's just a unary in
    # the transformed model.
    fmsgs = torch.cat([fmsg.view(batch_size, 1, -1) for fmsg in fmsgs], 1)
    bmsgs = torch.cat([bmsg.view(batch_size, 1, -1) for bmsg in bmsgs], 1)

    bb = bmsgs[:, 2:, :]
    ff = fmsgs[:, :-2, :]
    b0 = logsumexp(bb + ff, 2).view(batch_size, dim_node - 1)
    b1 = (
        logsumexp(bb[:, :, :-1] + ff[:, :, 1:], 2).view(batch_size, dim_node - 1)
        + node_potentials[:, :-1]
    )

    marginals = create_var(0, batch_size, dim_node)
    marginals[:, :-1] = torch.sigmoid(b1 - b0)

    # Could probably structure things so the Dth var doesn't need to be
    # special-cased.  but this will do for now.  rather than computing
    # a belief at a pairwise potential, we do it at the variable.
    b0_D = fmsgs[:, dim_node - 1, 0] + bmsgs[:, dim_node, 0]
    b1_D = fmsgs[:, dim_node - 1, 1] + bmsgs[:, dim_node, 1]
    marginals[:, dim_node - 1] = torch.sigmoid(b1_D - b0_D)

    return marginals


class TreeMarginals(object):
    r"""Perform marginal inference in models over spanning trees.
    The model considered is of the form:
    .. math::
        p(x) \propto \exp(\sum_{i=1}^m d_i x_i) \nu(x),
    where :math:`x` is a binary random vector with one coordinate per edge,
    and :math:`\nu(x)` is one if :math:`x` forms a spanning tree, or zero
    otherwise.
    The numbers :math:`d_i` are expected to be given by taking the upper
    triangular part of the adjacecny matrix. To extract the upper triangular
    part of a matrix, or to reconstruct them matrix from it, you can use the
    functions :py:meth:`~.triu` and :py:meth:`~.to_mat`.
    Arguments
    ---------
    n_vertices: int
      The number of vertices in the graph.
    cuda: bool
      Should the function work on cuda (on the current device) or cpu."""

    def __init__(self, n_vertices, cuda):
        self.n_vertices = n_vertices

        self.triu_mask = torch.triu(torch.ones(n_vertices, n_vertices), 1).byte()
        if cuda:
            self.triu_mask = self.triu_mask.cuda()

        n_edges = n_vertices * (n_vertices - 1) // 2
        # A is the edge incidence matrix, arbitrarily oriented.
        if cuda:
            A = torch.cuda.FloatTensor(n_vertices, n_edges)
        else:
            A = torch.FloatTensor(n_vertices, n_edges)
        A.zero_()

        k = 0
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                A[i, k] = +1
                A[j, k] = -1
                k += 1
        self.A = A[1:, :]  # We remove the first node from the matrix.

    def to_mat(self, triu):
        r"""Given the upper triangular part, reconstruct the matrix.
        Arguments
        ---------
        x: :class:`torch:torch.autograd.Variable`
            The upper triangular part, should be of size ``n * (n - 1) / 2``.
        Returns
        --------
        :class:`torch:torch.autograd.Variable`
          The ``(n, n)``-matrix whose upper triangular part filled in with
          ``x``, and the rest with zeroes"""
        if triu.is_cuda:
            matrix = torch.cuda.FloatTensor(self.n_vertices, self.n_vertices)
        else:
            matrix = torch.zeros(self.n_vertices, self.n_vertices)
        matrix.zero_()
        triu_mask = Variable(self.triu_mask, requires_grad=False)
        matrix = Variable(matrix, requires_grad=False)
        return matrix.masked_scatter(triu_mask, triu)

    def triu(self, matrix):
        r"""Given a matrix, extract its upper triangular part.
        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            A square matrix of size ``(n, n)``.
        Returns
        --------
        :class:`torch:torch.autograd.Variable`
          The upper triangular part of the given matrix, which is of size
          ``n * (n - 1) // 2``"""
        triu_mask = Variable(self.triu_mask, requires_grad=False)
        return torch.masked_select(matrix, triu_mask)

    def __call__(self, d):
        r"""Compute the marginals in the model.
        Arguments
        ---------
        d: :class:`torch:torch.autograd.Variable`
            A vector of size ``n * (n - 1) // 2`` containing the :math:`d_i`.
        Returns
        --------
        :class:`torch:torch.autograd.Variable`
            The marginal probabilities in a vector of size
            ``n * (n - 1) // 2``."""
        d = d - d.max()  # So that we don't have to compute large exponentials.

        # Construct the Laplacian.
        L_off = self.to_mat(torch.exp(d))
        L_off = L_off + L_off.t()
        L_dia = torch.diag(L_off.sum(1))
        L = L_dia - L_off
        L = L[1:, 1:]

        A = Variable(self.A, requires_grad=False)
        P = (1.0 / torch.diag(L)).view(1, -1)  # The diagonal pre-conditioner.
        Z, _ = torch.gesv(A, L * P.expand_as(L))
        Z = Z * P.t().expand_as(Z)
        # relu for numerical stability, the inside term should never be zero.
        return relu(torch.sum(Z * A, 0)) * torch.exp(d)


class MSTFn(Function):
    """Compute the minimum spanning tree given a matrix of pairwise weights."""

    @staticmethod
    def forward(ctx, weights):
        """Compute the MST given the edge weights.
        The behaviour is the same as that of ``minimum_spanning_tree` in
        ``scipy.sparse.csgraph``, namely i) the edges are assumed non-negative,
        ii) if ``weights[i, j]`` and ``weights[j, i]`` are both non-negative,
        their minimum is taken as the edge weight.
        Arguments
        ---------
        weights: :class:`torch:torch.Tensor`
            The adjacency matrix of size ``(n, n)``.
        Returns
        -------
        :class:`torch:torch.Tensor`
            An ``(n, n)`` matrix adjacency matrix of the minimum spanning tree.
            Indices corresponding to the edges in the MST are set to one, rest
            are set to zero.
            If both weights[i, j] and weights[j, i] are non-zero, then the one
            will be located in whichever holds the *smaller* value (ties broken
            arbitrarily).
        """
        mst_matrix = mst(weights.cpu().numpy()).toarray() > 0
        assert int(mst_matrix.sum()) + 1 == weights.size(0)
        return torch.Tensor(mst_matrix.astype(float))


class KSmallest(Function):
    """Return an indicator vector holing the smallest k elements in each row."""

    @staticmethod
    def forward(ctx, k, matrix):
        """Compute the positions holding the largest k elements in each row.
        Arguments
        ---------
        k: int
            How many elements to keep per row.
        matrix: :class:`torch:torch.Tensor`
            Tensor of size (n, m)
        Returns
        -------
        torch.Tensor of size (n, m)
           The positions that correspond to the k largest elements are set to
           one, the rest are set to zero."""
        ctx.mark_non_differentiable(matrix)
        matrix = matrix.numpy()
        indices = np.argsort(matrix, axis=1)
        mins = np.zeros_like(matrix)
        rows = np.arange(matrix.shape[0]).reshape(-1, 1)
        mins[rows, indices[:, :k]] = 1
        return torch.Tensor(mins)


class FRStatistic(object):
    """The classical Friedman-Rafsky test :cite:`friedman1979multivariate`.
    Arguments
    ----------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed Friedman-Rafsky test statistic.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.FRStatistic.pval`.
        Returns
        -------
        float
            The number of edges that do connect points from the *same* sample.
        """
        n_1 = sample_1.size(0)
        assert n_1 == self.n_1 and sample_2.size(0) == self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)

        mstf = MSTFn.apply
        mst_matrix = mstf(diffs)

        statistic = mst_matrix[:n_1, :n_1].sum() + mst_matrix[n_1:, n_1:].sum()

        if ret_matrix:
            return statistic, mst_matrix
        else:
            return statistic


class KNNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.
    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """

    def __init__(self, n_1, n_2, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.
        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)

        indices = (1.0 - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()

        for i in range(n):
            diffs[i, i] = float("inf")  # We don't want the diagonal selected.
        ksmallest = KSmallest.apply
        smallest = ksmallest(self.k, diffs.cpu())
        statistic = smallest[:n_1, :n_1].sum() + smallest[n_1:, n_1:].sum()

        if ret_matrix:
            return statistic, smallest
        else:
            return statistic


class SmoothFRStatistic(object):
    r"""The smoothed Friedman-Rafsky test :cite:`djolonga17graphtests`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample.
    cuda: bool
        If true, the arguments to :py:meth:`~.SmoothFRStatistic.__call__` must
        be be on the current cuda device. Otherwise, they should be on the cpu.
    """

    def __init__(self, n_1, n_2, cuda, compute_t_stat=True):
        n = n_1 + n_2
        self.n_1, self.n_2 = n_1, n_2
        # The idx_within tensor contains the indices that correspond to edges
        # that connect samples from within the same sample.
        # The matrix self.nbs is of size (n, n_edges) and has 1 in position
        # (i, j) if node i is incident to edge j. Specifically, note that
        # self.nbs @ mu will result in a vector that has at position i the sum
        # of the marginals of all edges incident to i, which we need in the
        # formula for the variance.
        if cuda:
            self.idx_within = torch.cuda.ByteTensor((n * (n - 1)) // 2)
            if compute_t_stat:
                self.nbs = torch.cuda.FloatTensor(n, self.idx_within.size()[0])
        else:
            self.idx_within = torch.ByteTensor((n * (n - 1)) // 2)
            if compute_t_stat:
                self.nbs = torch.FloatTensor(n, self.idx_within.size()[0])
        self.idx_within.zero_()
        if compute_t_stat:
            self.nbs.zero_()
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if compute_t_stat:
                    self.nbs[i, k] = 1
                    self.nbs[j, k] = 1
                if (i < n_1 and j < n_1) or (i >= n_1 and j >= n_1):
                    self.idx_within[k] = 1
                k += 1

        self.marginals_fn = TreeMarginals(n_1 + n_2, cuda)
        self.compute_t_stat = compute_t_stat

    def __call__(self, sample_1, sample_2, alphas, norm=2, ret_matrix=False):
        r"""Evaluate the smoothed Friedman-Rafsky test statistic.
        The test accepts several **inverse temperatures** in ``alphas``, does
        one test for each ``alpha``, and takes their mean as the statistic.
        Namely, using the notation in :cite:`djolonga17graphtests`, the
        value returned by this call if ``compute_t_stat=False`` is equal to:
        .. math::
            -\frac{1}{m}\sum_{j=m}^k T_{\pi^*}^{1/\alpha_j}(\textrm{sample}_1,
                                                            \textrm{sample}_2).
        If ``compute_t_stat=True``, the returned value is the t-statistic of
        the above quantity under the permutation null. Note that we compute the
        negated statistic of what is used in :cite:`djolonga17graphtests`, as
        it is exactly what we want to minimize when used as an objective for
        training implicit models.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, should be of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, should be of size ``(n_2, d)``.
        alphas: list of :class:`float` numbers
            The inverse temperatures.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.SmoothFRStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic, a t-statistic if ``compute_t_stat=True``.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        margs = None
        for alpha in alphas:
            margs_a = self.marginals_fn(self.marginals_fn.triu(-alpha * diffs))
            if margs is None:
                margs = margs_a
            else:
                margs = margs + margs_a

        margs = margs / len(alphas)
        idx_within = Variable(self.idx_within, requires_grad=False)
        n_1, n_2, n = self.n_1, self.n_2, self.n_1 + self.n_2
        m = margs.sum()
        t_stat = m - torch.masked_select(margs, idx_within).sum()
        if self.compute_t_stat:
            nbs = Variable(self.nbs, requires_grad=False)
            nbs_sum = (nbs.mm(margs.unsqueeze(1)) ** 2).sum()
            chi_1 = n_1 * n_2 / (n * (n - 1))
            chi_2 = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))
            var = (
                chi_1 * (1 - chi_2) * nbs_sum
                + chi_1 * chi_2 * (margs ** 2).sum()
                + chi_1 * (chi_2 - 4 * chi_1) * m ** 2
            )
            mean = 2 * m * n_1 * n_2 / (n * (n - 1))
            std = torch.sqrt(1e-5 + var)
        else:
            mean = 0.0
            std = 1.0

        if ret_matrix:
            return -(t_stat - mean) / std, margs
        else:
            return -(t_stat - mean) / std


class SmoothKNNStatistic(object):
    r"""The smoothed k-nearest neighbours test :cite:`djolonga17graphtests`.
    Note that the ``k=1`` case is computed directly using a SoftMax and should
    execute much faster than the statistics with ``k > 1``.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample.
    cuda: bool
        If true, the arguments to ``__call__`` must be be on the current
        cuda device. Otherwise, they should be on the cpu.
    k: int
        The number of nearest neighbours (k in kNN)."""

    def __init__(self, n_1, n_2, cuda, k, compute_t_stat=True):
        self.count_potential = torch.FloatTensor(1, k + 1)
        self.count_potential.fill_(NINF)
        self.count_potential[0, -1] = 0
        self.indices_cpu = (1 - torch.eye(n_1 + n_2)).byte()
        self.k = k
        self.n_1 = n_1
        self.n_2 = n_2
        self.cuda = cuda
        if cuda:
            self.indices = self.indices_cpu.cuda()
        else:
            self.indices = self.indices_cpu
        self.compute_t_stat = compute_t_stat

    def __call__(self, sample_1, sample_2, alphas, norm=2, ret_matrix=False):
        r"""Evaluate the smoothed kNN statistic.
        The test accepts several **inverse temperatures** in ``alphas``, does
        one test for each ``alpha``, and takes their mean as the statistic.
        Namely, using the notation in :cite:`djolonga17graphtests`, the
        value returned by this call if `compute_t_stat=False` is equal to:
        .. math::
            -\frac{1}{m}\sum_{j=m}^k T_{\pi^*}^{1/\alpha_j}(\textrm{sample}_1,
                                                            \textrm{sample}_2).
        If ``compute_t_stat=True``, the returned value is the t-statistic of
        the above quantity under the permutation null. Note that we compute the
        negated statistic of what is used in :cite:`djolonga17graphtests`, as
        it is exactly what we want to minimize when used as an objective for
        training implicit models.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alpha: list of :class:`float`
            The smoothing strengths.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.SmoothKNNStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic, a t-statistic if ``compute_t_stat=True``.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1
        assert n_2 == self.n_2
        n = n_1 + n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12)
        indices = Variable(self.indices, requires_grad=False)
        indices_cpu = Variable(self.indices_cpu, requires_grad=False)
        k = self.count_potential.size()[1] - 1
        assert k == self.k
        count_potential = Variable(
            self.count_potential.expand(n, k + 1), requires_grad=False
        )

        diffs = torch.masked_select(diffs, indices).view(n, n - 1)

        margs_ = None
        for alpha in alphas:
            if self.k == 1:
                margs_a = softmax(-alpha * diffs, dim=1)
            else:
                margs_a = inference_cardinality(-alpha * diffs.cpu(), count_potential)
            if margs_ is None:
                margs_ = margs_a
            else:
                margs_ = margs_ + margs_a

        margs_ /= len(alphas)
        # The variable margs_ is a matrix of size n x n-1, which we want to
        # reshape to n x n by adding a zero diagonal, as it makes the following
        # logic easier to follow. The variable margs_ is on the GPU when k=1.
        if margs_.is_cuda:
            margs = torch.cuda.FloatTensor(n, n)
        else:
            margs = torch.FloatTensor(n, n)
        margs.zero_()
        margs = Variable(margs, requires_grad=False)
        if margs_.is_cuda:
            margs.masked_scatter_(indices, margs_.view(-1))
        else:
            margs.masked_scatter_(indices_cpu, margs_.view(-1))

        t_stat = margs[:n_1, n_1:].sum() + margs[n_1:, :n_1].sum()
        if self.compute_t_stat:
            m = margs.sum()
            mean = 2 * m * n_1 * n_2 / (n * (n - 1))
            nbs_sum = ((margs.sum(0).view(-1) + margs.sum(1).view(-1)) ** 2).sum()
            flip_sum = (margs * margs.transpose(1, 0)).sum()
            chi_1 = n_1 * n_2 / (n * (n - 1))
            chi_2 = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))
            var = (
                chi_1 * (1 - chi_2) * nbs_sum
                + chi_1 * chi_2 * (margs ** 2).sum()
                + chi_1 * chi_2 * flip_sum
                + chi_1 * (chi_2 - 4 * chi_1) * m ** 2
            )
            std = torch.sqrt(1e-5 + var)
        else:
            mean = 0.0
            std = 1.0

        if ret_matrix:
            return -(t_stat - mean) / std, margs
        else:
            return -(t_stat - mean) / std


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1.0 / (n_1 * (n_1 - 1))
        self.a11 = 1.0 / (n_2 * (n_2 - 1))
        self.a01 = -1.0 / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(-alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[: self.n_1, : self.n_1]
        k_2 = kernels[self.n_1 :, self.n_2 :]
        k_12 = kernels[: self.n_1, self.n_2 :]

        mmd = (
            2 * self.a01 * k_12.sum()
            + self.a00 * (k_1.sum() - torch.trace(k_1))
            + self.a11 * (k_2.sum() - torch.trace(k_2))
        )

        if ret_matrix:
            return mmd, kernels
        else:
            return mmd


class EnergyStatistic:
    r"""The energy test of :cite:`szekely2013energy`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = -1.0 / (n_1 * n_1)
        self.a11 = -1.0 / (n_2 * n_2)
        self.a01 = 1.0 / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, ret_matrix=False):
        r"""Evaluate the statistic.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.EnergyStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)
        d_1 = distances[: self.n_1, : self.n_1].sum()
        d_2 = distances[-self.n_2 :, -self.n_2 :].sum()
        d_12 = distances[: self.n_1, -self.n_2 :].sum()

        loss = 2 * self.a01 * d_12 + self.a00 * d_1 + self.a11 * d_2

        if ret_matrix:
            return loss, distances
        else:
            return loss
