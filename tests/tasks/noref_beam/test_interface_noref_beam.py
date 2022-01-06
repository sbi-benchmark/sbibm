import numpy as np
import pytest
import torch
from pyro import distributions as pdist
from pyro import util as putil

import sbibm
from sbibm.algorithms import rej_abc
from sbibm.algorithms.sbi.snle import run as run_snle
from sbibm.algorithms.sbi.snpe import run as run_snpe
from sbibm.metrics import c2st
from sbibm.metrics.ppc import median_distance

putil.set_rng_seed(47)


def test_no_reference_posterior():

    task = sbibm.get_task("noref_beam")

    with pytest.raises(FileNotFoundError):
        reference_samples = task.get_reference_posterior_samples(num_observation=1)


################################################
## sbibm compliant API tests as documented in
## the top-level README.md of sbibm


def test_quick_demo_rej_abc():

    task = sbibm.get_task("noref_beam")
    posterior_samples, _, _ = rej_abc(
        task=task, num_samples=50, num_observation=1, num_simulations=500
    )

    assert posterior_samples != None
    assert posterior_samples.shape[0] == 50


def test_benchmark_snpe_metrics_priorsim():

    task = sbibm.get_task("noref_beam")

    nobs = 1
    theta_o = task.get_prior()(num_samples=nobs)
    sim = task.get_simulator()
    x_o = sim(theta_o)

    assert x_o.shape[-1] == 400
    assert task.dim_data == 400

    outputs, nsim, logprob_truep = run_snpe(
        task,
        observation=x_o,
        num_samples=1,
        num_simulations=64,
        neural_net="mdn",
        hidden_features=4,
        simulation_batch_size=32,
        training_batch_size=32,
        num_rounds=1,  # let's do NPE not SNPE (to avoid MCMC)
        max_num_epochs=30,
    )

    assert outputs.shape
    assert outputs.shape[0] > 0
    assert logprob_truep == None

    predictive_samples = sim(outputs)
    value = median_distance(predictive_samples, x_o)

    assert value > 0


def test_benchmark_snpe_selfobserved_single_mediandist():

    task = sbibm.get_task("noref_beam")

    nobs = 1
    theta_o = task.get_prior()(num_samples=nobs)

    assert theta_o.shape == (nobs, 4)

    sim = task.get_simulator()
    x_o = sim(theta_o)

    assert x_o.shape[-1] == 400
    assert x_o.shape[0] == nobs
    assert task.dim_data == 400

    outputs, nsim, logprob_truep = run_snpe(
        task,
        observation=x_o,
        num_samples=8,
        num_simulations=64,
        neural_net="mdn",
        hidden_features=4,
        simulation_batch_size=32,
        training_batch_size=32,
        num_rounds=1,  # let's do NPE not SNPE (to avoid MCMC)
        max_num_epochs=30,
    )

    assert outputs.shape
    assert outputs.shape[0] > 0
    assert logprob_truep == None

    predictive_samples = sim(outputs)
    value = median_distance(predictive_samples, x_o)

    assert value > 0


def test_benchmark_snpe_multiobservations_raises():

    task = sbibm.get_task("noref_beam")

    nobs = 4
    theta_o = task.get_prior()(num_samples=nobs)

    assert theta_o.shape == (nobs, 4)

    sim = task.get_simulator()
    x_o = sim(theta_o)

    assert x_o.shape[-1] == 400
    assert x_o.shape[0] == nobs
    assert task.dim_data == 400

    # remind me in case this behavior changes
    with pytest.raises(ValueError) as ve:
        outputs, nsim, logprob_truep = run_snpe(
            task,
            observation=x_o,
            num_samples=8,
            num_simulations=64,
            neural_net="mdn",
            hidden_features=4,
            simulation_batch_size=32,
            training_batch_size=32,
            num_rounds=1,  # let's do NPE not SNPE (to avoid MCMC)
            max_num_epochs=30,
        )


def test_benchmark_metrics_snle_multiobservations():

    task = sbibm.get_task("noref_beam")

    nobs = 4
    theta_o = task.get_prior()(num_samples=nobs)

    assert theta_o.shape == (nobs, 4)

    sim = task.get_simulator()
    x_o = sim(theta_o)

    assert x_o.shape[-1] == 400
    assert x_o.shape[0] == nobs
    assert task.dim_data == 400

    nsamples_gen = 16
    # remind me in case this behavior changes
    outputs, nsim, logprob_truep = run_snle(
        task,
        observation=x_o,
        num_samples=nsamples_gen,
        num_simulations=64,
        simulation_batch_size=32,
        training_batch_size=32,
        num_rounds=1,  # let's do NLE not SNLE
        max_num_epochs=50,
        mcmc_parameters={
            # "num_chains": 1,
            # "warmup_steps": 5,  #
            "thin": 3
        },
    )

    assert outputs.shape
    assert outputs.shape[0] > 0
    assert outputs.shape == (nsamples_gen, theta_o.shape[-1])
    assert logprob_truep == None

    predictive_samples = sim(outputs)

    assert predictive_samples.shape != x_o.shape

    obs = predictive_samples[0 : x_o.shape[0], ...]

    assert obs.shape == x_o.shape

    value = median_distance(obs, x_o)

    assert value > 0
