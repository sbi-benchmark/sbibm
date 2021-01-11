from typing import Any

import torch
from tqdm.auto import tqdm

from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Runtime baseline

    Draws `num_simulations` samples from prior and simulates, discards outcomes,
    returns tensor of NaNs.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        batch_size: Batch size for simulations

    Returns:
        Random samples from prior
    """
    prior = task.get_prior()
    simulator = task.get_simulator()

    batch_size = min(batch_size, num_simulations)
    num_batches = int(num_simulations / batch_size)

    for i in tqdm(range(num_batches)):
        _ = simulator(prior(num_samples=batch_size))

    assert simulator.num_simulations == num_simulations

    samples = float("nan") * torch.ones((num_samples, task.dim_parameters))

    return samples
