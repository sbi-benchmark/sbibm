import numpy as np
import pytest
import torch

import sbibm
from sbibm.algorithms.abcpy.rejection_abc import run as rej_abc


@pytest.mark.parametrize(
    "task_name",
    [(task_name) for task_name in ["gaussian_linear_uniform"]],
)
def test_abcpy_rejection(
    task_name,
    num_observation=1,
    num_samples=10_000,
    num_simulations=1_000,
):
    # set random seeds:
    torch.manual_seed(1)
    np.random.seed(1)

    task = sbibm.get_task(task_name)

    posterior_samples, actual_num_simulations, _ = rej_abc(
        task=task,
        num_samples=num_samples,
        num_observation=num_observation,
        num_simulations=num_simulations,
        quantile=0.1,
        num_top_samples=None,
        kde_bandwidth="cv",
    )
    assert actual_num_simulations == num_simulations
    assert posterior_samples.shape[0] == num_samples

    posterior_samples, actual_num_simulations, _ = rej_abc(
        task=task,
        num_samples=num_samples,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_top_samples=500,
        kde_bandwidth="cv",
    )
    assert actual_num_simulations == num_simulations
    assert posterior_samples.shape[0] == num_samples

    # test multiple simulations per parameter
    posterior_samples, actual_num_simulations, _ = rej_abc(
        task=task,
        num_samples=num_samples,
        num_observation=num_observation,
        num_simulations_per_param=2,
        num_simulations=num_simulations,
        eps=6,
        num_top_samples=None,
        kde_bandwidth="cv",
    )
    assert actual_num_simulations == num_simulations
    assert posterior_samples.shape[0] == num_samples
