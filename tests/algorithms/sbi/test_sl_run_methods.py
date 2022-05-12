import pytest
import torch

import sbibm
from sbibm.algorithms.sbi.sl import run as run_sl


@pytest.mark.parametrize("task_name", ("gaussian_linear",))
@pytest.mark.parametrize("run_method", (run_sl,))
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
