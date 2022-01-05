import pytest
import torch

import sbibm
from sbibm.algorithms.sbi.mcabc import run as run_mcabc
from sbibm.algorithms.sbi.smcabc import run as run_smcabc
from sbibm.algorithms.sbi.mcabc import build_posterior as build_posterior_mcabc
from sbibm.algorithms.sbi.smcabc import build_posterior as build_posterior_smcabc
from sbibm.metrics.c2st import c2st


@pytest.mark.parametrize("task_name", ("gaussian_linear", "two_moons"))
@pytest.mark.parametrize(
    "build_method", (build_posterior_mcabc, build_posterior_smcabc)
)
def test_build_posterior_interface(
    task_name,
    build_method,
    num_simulations=100,
    num_samples=5,
):
    task = sbibm.get_task(task_name)
    nobs = 3
    post, summary = build_method(
        task=task,
        num_simulations=num_simulations,
        num_observation=nobs,
        num_samples=num_samples,
    )

    assert len(summary) != 0
    assert len(list(summary.keys())) > 0

    assert hasattr(post, "sample")
    assert hasattr(post, "log_prob")

    tp_exp = task.get_true_parameters(num_observation=nobs)
    tp_obs = task.get_true_parameters(num_observation=nobs + 1)

    assert not torch.allclose(tp_exp, tp_obs)

    logprob_exp = post.log_prob(tp_exp)
    logprob_obs = post.log_prob(tp_obs)

    assert logprob_exp.shape == logprob_obs.shape
    assert not torch.allclose(logprob_exp, logprob_obs)


@pytest.mark.parametrize("task_name", ("gaussian_linear", "two_moons"))
@pytest.mark.parametrize("run_method", (run_mcabc, run_smcabc))
def test_run_posterior_interface(
    task_name,
    run_method,
    num_simulations=100,
    num_samples=5,
):
    task = sbibm.get_task(task_name)

    samples, _, _ = run_method(
        task=task,
        num_simulations=num_simulations,
        num_observation=3,
        num_samples=num_samples,
    )

    # we are not interested in testing for correctness
    assert samples.shape == torch.Size([num_samples, task.dim_parameters])
