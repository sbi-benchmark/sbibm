from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyro
import scipy.stats as stats
import torch
import torch.distributions.transforms as transforms

import sbibm
import sbibm.third_party.kgof as kgof
import sbibm.third_party.kgof.data as data
import sbibm.third_party.kgof.density as density
import sbibm.third_party.kgof.goftest as gof
import sbibm.third_party.kgof.kernel as kernel
import sbibm.third_party.kgof.util as util
from sbibm.third_party.kgof.density import UnnormalizedDensity


def test_fssd():
    """Test FSSD with Gaussian kernel (median heuristic) and randomized test locations

    Following the example in:
    https://github.com/wittawatj/kernel-gof/blob/master/kgof/ex/ex1_vary_n.py
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

    # FSSD
    J = 10
    null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=seed)
    # Fit a multivariate normal to the data X (n x d) and draw J points from the fit.
    V = util.fit_gaussian_draw(samples.data(), J=J, seed=seed + 1)
    fssd_med = gof.FSSD(p, k, V, null_sim=null_sim, alpha=0.01)
    test_result = fssd_med.perform_test(samples)
    print(test_result)
    assert test_result["h0_rejected"] == False

    # FSSD with samples from different density
    J = 10  # Fails with J=8, passes with J=10 (chance)
    ds = data.DSLaplace(d=d, loc=0, scale=1.0 / np.sqrt(2))
    samples = ds.sample(n, seed=seed + 1)
    sig2 = util.meddistance(samples.data(), subsample=1000) ** 2
    # NOTE: Works much better with the bandwidth that was optimized under FSSD:
    # sig2 = 0.3228712361986835
    k = kernel.KGauss(sig2)
    print(f"Kernel bandwidth: {sig2}")
    null_sim = gof.FSSDH0SimCovObs(n_simulate=3000, seed=seed)
    # TODO: is this what we want if samples come from another distribution ?!
    V = util.fit_gaussian_draw(samples.data(), J=J, seed=seed + 1)
    fssd_med = gof.FSSD(p, k, V, null_sim=null_sim, alpha=0.01)
    test_result = fssd_med.perform_test(samples)
    print(test_result)
    assert test_result["h0_rejected"] == True


def test_fssd_opt():
    """Test FSSD with optimized test locations

    Following the example in:
    https://github.com/wittawatj/kernel-gof/blob/master/ipynb/demo_kgof.ipynb
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

    # Split dataset
    tr, te = samples.split_tr_te(tr_proportion=0.2, seed=2)

    # Optimization
    opts = {
        "reg": 1e-2,  # regularization parameter in the optimization objective
        "max_iter": 50,  # maximum number of gradient ascent iterations
        "tol_fun": 1e-7,  # termination tolerance of the objective
    }
    # J is the number of test locations (or features). Typically not larger than 10
    J = 1
    V_opt, gw_opt, opt_info = gof.GaussFSSD.optimize_auto_init(p, tr, J, **opts)
    print(V_opt)
    print(f"Kernel bandwidth: {gw_opt}")
    print(opt_info)

    # FSSD
    fssd_opt = gof.GaussFSSD(p, gw_opt, V_opt, alpha=0.01)
    test_result = fssd_opt.perform_test(te)
    test_result
    print(test_result)
    assert test_result["h0_rejected"] == False

    # FSSD with samples from different density
    ds = data.DSLaplace(d=d, loc=0, scale=1.0 / np.sqrt(2))
    samples = ds.sample(n, seed=seed + 1)
    tr, te = samples.split_tr_te(tr_proportion=0.2, seed=2)
    opts = {
        "reg": 1e-2,  # regularization parameter in the optimization objective
        "max_iter": 50,  # maximum number of gradient ascent iterations
        "tol_fun": 1e-7,  # termination tolerance of the objective
    }
    J = 1  # J is the number of test locations (or features)
    V_opt, gw_opt, opt_info = gof.GaussFSSD.optimize_auto_init(p, tr, J, **opts)
    print(f"Kernel bandwidth: {gw_opt}")

    # FSSD
    fssd_opt = gof.GaussFSSD(p, gw_opt, V_opt, alpha=0.01)
    test_result = fssd_opt.perform_test(te)
    print(test_result)
    assert test_result["h0_rejected"] == True
