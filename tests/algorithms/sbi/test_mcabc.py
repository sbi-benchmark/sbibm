import pytest
import torch

import sbibm
from sbibm.algorithms.sbi.mcabc import run as run_posterior
from sbibm.metrics.c2st import c2st


@pytest.mark.parametrize(
    "task_name",
    [
        task_name
        for task_name in [
            "gaussian_linear",
            "two_moons",
        ]
    ],
)
def test_run_posterior_interface(
    task_name,
    num_simulations=100,
    num_samples=5,
):
    task = sbibm.get_task(task_name)

    samples = run_posterior(
        task=task,
        num_simulations=num_simulations,
        num_observation=3,
        num_samples=num_samples,
    )

    # we are not interested in testing for correctness
    assert len(samples.shape) > 0
