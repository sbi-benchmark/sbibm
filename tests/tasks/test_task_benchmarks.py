import re

import pytest
import torch

from sbibm import get_available_tasks, get_task
from sbibm.algorithms.sbi.snpe import run
from sbibm.metrics.ppc import median_distance

# maybe use the pyro facilities
torch.manual_seed(47)

# ################################################
# ## demonstrate on how to run a minimal benchmark
# ## see https://github.com/sbi-benchmark/results/blob/main/benchmarking_sbi/run.py


@pytest.mark.parametrize(
    "task_name",
    [tn for tn in get_available_tasks() if not re.search("noref|lotka|sir", tn)],
)
def test_benchmark_metrics_selfobserved(task_name):

    task = get_task(task_name)

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


# def test_benchmark_metrics():

#     from sbibm.algorithms.sbi.snpe import run
#     from sbibm.metrics.ppc import median_distance

#     task = get_task("two_moons")
#     sim = task.get_simulator()

#     outputs, nsim, logprob_truep = run(
#         task,
#         num_observation=7,
#         num_samples=64,
#         num_simulations=100,
#         neural_net="mdn",
#         num_rounds=1,  # let's do NPE not SNPE (to avoid MCMC)
#     )

#     assert outputs.shape
#     assert outputs.shape[0] > 0
#     assert logprob_truep == None

#     predictive_samples = sim(outputs)
#     x_o = task.get_observation(7)
#     value = median_distance(predictive_samples, x_o)

#     assert value > 0
