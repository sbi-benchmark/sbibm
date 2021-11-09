import pytest
import torch

import sbibm
from sbibm.tasks.two_moons.task import TwoMoons

torch.manual_seed(47)


def test_task_constructs():

    t = TwoMoons()

    assert t


def test_obtain_task():

    task = sbibm.get_task("two_moons")

    assert task is not None


def test_obtain_prior():

    task = sbibm.get_task("two_moons")  # See sbibm.get_available_tasks() for all tasks
    prior = task.get_prior()

    assert prior is not None


def test_obtain_simulator():

    task = sbibm.get_task("two_moons")

    simulator = task.get_simulator()

    assert simulator is not None


def test_observe_once():

    task = sbibm.get_task("two_moons")

    x_o = task.get_observation(num_observation=1)

    assert x_o is not None
    assert hasattr(x_o, "shape")


def test_obtain_prior_samples():

    task = sbibm.get_task("two_moons")
    prior = task.get_prior()
    nsamples = 10

    thetas = prior(num_samples=nsamples)

    assert thetas.shape == (nsamples, 2)


def test_simulate_from_thetas():

    task = sbibm.get_task("two_moons")
    prior = task.get_prior()
    sim = task.get_simulator()
    nsamples = 10

    thetas = prior(num_samples=nsamples)
    xs = sim(thetas)

    assert xs.shape == (nsamples, 2)


def test_reference_posterior_exists():

    task = sbibm.get_task("two_moons")

    reference_samples = task.get_reference_posterior_samples(num_observation=1)

    assert hasattr(reference_samples, "shape")
    assert len(reference_samples.shape) == 2
    assert reference_samples.shape == (10_000, 2)


# @pytest.fixture
# def vanilla_samples():

#     task = sbibm.get_task("two_moons")
#     prior = task.get_prior()
#     sim = task.get_simulator()
#     nsamples = 1_000

#     thetas = prior(num_samples=nsamples)
#     xs = sim(thetas)

#     return task, thetas, xs


def test_quick_demo_rej_abc():

    from sbibm.algorithms import rej_abc  # See help(rej_abc) for keywords

    task = sbibm.get_task("two_moons")
    posterior_samples, _, _ = rej_abc(
        task=task, num_samples=50, num_observation=1, num_simulations=500
    )

    assert posterior_samples != None
    assert posterior_samples.shape[0] == 50


def test_quick_demo_c2st():

    from sbibm.algorithms import rej_abc  # See help(rej_abc) for keywords

    task = sbibm.get_task("two_moons")
    posterior_samples, _, _ = rej_abc(
        task=task, num_samples=50, num_observation=1, num_simulations=500
    )

    from sbibm.metrics import c2st

    reference_samples = task.get_reference_posterior_samples(num_observation=1)
    c2st_accuracy = c2st(reference_samples, posterior_samples)

    assert c2st_accuracy > 0.0
    assert c2st_accuracy < 1.0


################################################
## demonstrate on how to run a minimal benchmark
## see https://github.com/sbi-benchmark/results/blob/main/benchmarking_sbi/run.py


def test_benchmark_metrics_selfobserved():

    from sbibm.algorithms.sbi.snpe import run
    from sbibm.metrics.ppc import median_distance

    task = sbibm.get_task("two_moons")

    nobs = 1
    theta_o = task.get_prior()(num_samples=nobs)
    sim = task.get_simulator()
    x_o = sim(theta_o)

    outputs, nsim, logprob_truep = run(
        task,
        observation=x_o,
        num_samples=16,
        num_simulations=64,
        neural_net="mdn",
        num_rounds=1,  # let's do NPE not SNPE (to avoid MCMC)
    )

    assert outputs.shape
    assert outputs.shape[0] > 0
    assert logprob_truep == None

    predictive_samples = sim(outputs)
    value = median_distance(predictive_samples, x_o)

    assert value > 0
    assert value > 0.5


def test_benchmark_metrics():

    from sbibm.algorithms.sbi.snpe import run
    from sbibm.metrics.ppc import median_distance

    task = sbibm.get_task("two_moons")
    sim = task.get_simulator()

    outputs, nsim, logprob_truep = run(
        task,
        num_observation=7,
        num_samples=64,
        num_simulations=100,
        neural_net="mdn",
        num_rounds=1,  # let's do NPE not SNPE (to avoid MCMC)
    )

    assert outputs.shape
    assert outputs.shape[0] > 0
    assert logprob_truep == None

    predictive_samples = sim(outputs)
    x_o = task.get_observation(7)
    value = median_distance(predictive_samples, x_o)

    assert value > 0
