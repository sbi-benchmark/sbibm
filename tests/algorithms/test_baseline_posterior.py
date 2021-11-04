import pytest
import torch

import sbibm
from sbibm.algorithms.pytorch.baseline_posterior import run as run_posterior
from sbibm.metrics.c2st import c2st


@pytest.mark.parametrize(
    "task_name,num_observation",
    [
        (task_name, num_observation)
        for task_name in [
            "gaussian_linear",
            "gaussian_linear_uniform",
            "slcp",
        ]
        for num_observation in range(1, 11)
    ],
)
def test_posterior(
    task_name,
    num_observation,
    num_samples=10_000,
):
    task = sbibm.get_task(task_name)

    samples = run_posterior(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        rerun=True,
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    acc = c2st(samples, reference_samples[:num_samples, :])

    assert torch.abs(acc - 0.5) < 0.025
