import itertools
import logging
import time

import numpy as np
import torch

from sbibm.third_party.igms.main import ExpQuadKernel as tp_ExpQuadKernel
from sbibm.third_party.igms.main import mmd2_unbiased as tp_mmd2_unbiased
from sbibm.third_party.torch_two_sample.main import MMDStatistic as tp_MMDStatistic
from sbibm.utils.torch import get_default_device

log = logging.getLogger(__name__)


def mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    implementation: str = "tp_sutherland",
    z_score: bool = False,
    bandwidth: str = "X",
) -> torch.Tensor:
    """Estimate MMD^2 statistic with Gaussian kernel

    Currently different implementations are available, in order to validate accuracy and compare speeds. The widely used median heuristic for bandwidth-selection of the Gaussian kernel is used.
    """
    if torch.isnan(X).any() or torch.isnan(Y).any():
        return torch.tensor(float("nan"))

    tic = time.time()  # noqa

    if z_score:
        X_mean = torch.mean(X, axis=0)
        X_std = torch.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    n_1 = X.shape[0]
    n_2 = Y.shape[0]

    # Bandwidth
    if bandwidth == "X":
        sigma_tensor = torch.median(torch.pdist(X))
    elif bandwidth == "XY":
        sigma_tensor = torch.median(torch.pdist(torch.cat([X, Y])))
    else:
        raise NotImplementedError

    # Compute MMD
    if implementation == "tp_sutherland":
        K = tp_ExpQuadKernel(X, Y, sigma=sigma_tensor)
        statistic = tp_mmd2_unbiased(K)

    elif implementation == "tp_djolonga":
        alpha = 1 / (2 * sigma_tensor ** 2)
        test = tp_MMDStatistic(n_1, n_2)
        statistic = test(X, Y, [alpha])

    else:
        raise NotImplementedError

    toc = time.time()  # noqa
    # log.info(f"Took {toc-tic:.3f}sec")

    return statistic
