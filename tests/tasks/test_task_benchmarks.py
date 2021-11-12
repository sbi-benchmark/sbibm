import re

import pyro
import pytest
import torch

from sbibm import get_available_tasks, get_task
from sbibm.algorithms.sbi.snpe import run
from sbibm.metrics.ppc import median_distance

pyro.util.set_rng_seed(47)

# ################################################
# ## demonstrate on how to run a minimal benchmark
# ## see https://github.com/sbi-benchmark/results/blob/main/benchmarking_sbi/run.py


@pytest.mark.parametrize(
    "task_name",
    [tn for tn in get_available_tasks() if not re.search("lotka|sir", tn)],
)
def test_benchmark_metrics_selfobserved(task_name):

    task = get_task(task_name)

    nobs = 1  # maybe randomly dice this?
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
