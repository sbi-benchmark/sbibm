import pytest
import torch

import sbibm
from sbibm.algorithms.sbi.mcabc import run as run_mcabc
from sbibm.algorithms.sbi.smcabc import run as run_smcabc
from sbibm.metrics.c2st import c2st


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
