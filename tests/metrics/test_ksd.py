from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyro
import scipy.stats as stats
import torch
import torch.distributions.transforms as transforms

import sbibm
import sbibm.third_party.kgof
import sbibm.third_party.kgof.data as data
import sbibm.third_party.kgof.density as density
import sbibm.third_party.kgof.goftest as gof
import sbibm.third_party.kgof.kernel as kernel
import sbibm.third_party.kgof.util as util
from sbibm.third_party.kgof.density import UnnormalizedDensity


def test_ksd():
    """Test quadratic time KSD

    Following the example in:
    https://github.com/wittawatj/kernel-gof/blob/master/ipynb/gof_kernel_stein.ipynb
    """
    seed = 42

    d = 2  # dimensionality
    n = 800  # samples

    # Density
    mean = np.zeros(d)
    variance = 1.0
    p = density.IsotropicNormal(mean, variance)

    # Samples from same density
    ds = data.DSIsotropicNormal(mean, variance)
    samples = ds.sample(n, seed=seed + 1)

    # Gaussian kernel with median heuristic
    sig2 = util.meddistance(samples.data(), subsample=1000) ** 2
    k = kernel.KGauss(sig2)
    print(f"Kernel bandwidth: {sig2}")

    # KSD
    bootstrapper = gof.bootstrapper_rademacher
    kstein = gof.KernelSteinTest(
        p, k, bootstrapper=bootstrapper, alpha=0.01, n_simulate=500, seed=seed + 1
    )
    test_result = kstein.perform_test(
        samples, return_simulated_stats=False, return_ustat_gram=False
    )
    print(test_result)
    assert test_result["h0_rejected"] == False

    # KSD with samples from different density
    ds = data.DSLaplace(d=d, loc=0, scale=1.0 / np.sqrt(2))
    samples = ds.sample(n, seed=seed + 1)
    sig2 = util.meddistance(samples.data(), subsample=1000) ** 2
    print(f"Kernel bandwidth: {sig2}")
    k = kernel.KGauss(sig2)
    bootstrapper = gof.bootstrapper_rademacher
    kstein = gof.KernelSteinTest(
        p, k, bootstrapper=bootstrapper, alpha=0.01, n_simulate=500, seed=seed + 1
    )
    test_result = kstein.perform_test(
        samples, return_simulated_stats=False, return_ustat_gram=False
    )
    print(test_result)
    assert test_result["h0_rejected"] == True
