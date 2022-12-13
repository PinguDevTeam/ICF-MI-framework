""" This file contains the utility functions for running DaMAT pack"""

import numpy as np


def ttsvd(data, dataNorm, eps=0.1, dtype=np.float32):
    """
    Computes Tensor-Train decomposition/approximation of tensors using `TTSVD`_
    algorithm.

    Parameters
    ----------
    data:obj:`numpy.array`
        Tensor to be decomposed/approximated.
    dataNorm:obj:`float`
        Norm of the tensor. This parameter is used to determine the truncation bound.
    eps:obj:`float`, optional
        Relative error upper bound for TT-decomposition. Set to 0.1 by default.
    dtype:obj:`type`, optional
        Data type to be used during computations. Set to `np.float32` by default .

    Returns
    -------
    ranks:obj:`list`
        List of TT-ranks.
    cores:obj:`numpy.ndarray`
        Cores of the TT-approximation.

    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286
    """
    inputShape = data.shape
    dimensions = len(data.shape)
    delta = (eps / ((dimensions - 1) ** (0.5))) * dataNorm
    ranks = [1]
    cores = []
    for k in range(dimensions - 1):
        nk = inputShape[k]
        data = data.reshape(
            ranks[k] * nk, int(np.prod(data.shape) / (ranks[k] * nk)), order="F"
        ).astype(dtype)
        u, s, v = np.linalg.svd(data, False, True)
        slist = list(s * s)
        slist.reverse()
        truncpost = [
            idx for idx, element in enumerate(np.cumsum(slist)) if element <= delta**2
        ]
        ranks.append(len(s) - len(truncpost))

        u = u[:, : ranks[-1]]

        cores.append(u.reshape(ranks[k], nk, ranks[k + 1], order="F"))
        data = np.zeros_like(v[: ranks[-1], :])
        for idx, sigma in enumerate(s[: ranks[-1]]):
            data[idx, :] = sigma * v[idx, :]

    ranks.append(1)
    cores.append(data.reshape(ranks[-2], inputShape[-1], ranks[-1], order="F"))
    return ranks, cores
