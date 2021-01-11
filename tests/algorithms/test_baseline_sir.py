import pytest
import torch
from sbi.utils.plot import pairplot

import sbibm
from sbibm.algorithms.pytorch.baseline_sir import run
from sbibm.algorithms.pytorch.utils.proposal import get_proposal
from sbibm.metrics.c2st import c2st


def test_sir_with_proposal(
    plt,
    task_name="gaussian_linear_uniform",
    num_observation=1,
    num_samples=10000,
    num_simulations=10_000_000,
    batch_size=10000,
    prior_weight=0.01,
):
    task = sbibm.get_task(task_name)

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    proposal_dist = get_proposal(
        task=task,
        samples=reference_samples,
        prior_weight=prior_weight,
        bounded=False,
        density_estimator="flow",
        flow_model="nsf",
    )

    samples = run(
        task=task,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_samples=num_samples,
        batch_size=batch_size,
        proposal_dist=proposal_dist,
    )

    num_samples_plotting = 1000
    pairplot(
        [
            samples.numpy()[:num_samples_plotting, :],
            reference_samples.numpy()[:num_samples_plotting, :],
        ]
    )

    acc = c2st(samples, reference_samples[:num_samples, :])

    assert torch.abs(acc - 0.5) < 0.01
