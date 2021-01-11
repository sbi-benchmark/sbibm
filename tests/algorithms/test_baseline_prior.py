import pytest
import torch

import sbibm
from sbibm.algorithms.pytorch.baseline_prior import run as run_prior
from sbibm.metrics.c2st import c2st


@pytest.mark.parametrize(
    "task_name",
    [(task_name) for task_name in ["gaussian_linear"]],
)
def test_prior(
    task_name,
    num_observation=1,
    num_samples=1000,
):
    task = sbibm.get_task(task_name)

    samples = run_prior(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
    )

    assert len(samples) == num_samples
