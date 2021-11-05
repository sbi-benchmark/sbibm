import pytest
import torch
import random

import sbibm
from sbibm.algorithms.sbi.snpe import run as run_posterior
from sbibm.metrics.c2st import c2st

#a fast test
@pytest.mark.parametrize(
    "task_name,num_observation",
    [
        (task_name, num_observation)
        for task_name in ["gaussian_linear", "slcp",]
        for num_observation in random.sample(range(1, 11), 2)
    ],
)
def test_npe_posterior(
    task_name, num_observation, num_simulations=1_000, num_samples=100
):
    task = sbibm.get_task(task_name)

    samples = run_posterior(
        task=task,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_samples=num_samples,
        num_rounds=1,
        neural_net="mdn" #fast test
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    acc = c2st(samples, reference_samples[:num_samples, :])

    assert torch.abs(acc - 0.5) < 0.025
