import math
import time
from typing import Any, Optional

import torch
from tqdm.auto import tqdm

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.torch import choice


def run(
    task: Task,
    num_samples: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_simulations: Optional[int] = None,
    low: Optional[torch.Tensor] = None,
    high: Optional[torch.Tensor] = None,
    eps: float = 0.001,
    resolution: Optional[int] = None,
    batch_size: int = 10000,
    save: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Random samples from gridded posterior as baseline

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_simulations: Number of simulations to determine resolution
        low: Lower limit per dimension, tries to infer it if not passed
        high: Upper limit per dimension, tries to infer it if not passed
        eps: Eps added to bounds to avoid NaN evaluations
        resolution: Resolution for all dimensions, alternatively use `num_simulations`
        batch_size: Batch size
        save: If True, saves grid and log probs

    Returns:
        Random samples from reference posterior
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    assert not (num_simulations is None and resolution is None)
    assert not (num_simulations is not None and resolution is not None)

    tic = time.time()
    log = sbibm.get_logger(__name__)

    if num_simulations is not None:
        resolution = int(
            math.floor(math.exp(math.log(num_simulations) / task.dim_parameters))
        )
    log.info(f"Resolution: {resolution}")

    # Infer bounds if not passed
    prior_params = task.get_prior_params()
    if low is None:
        if "low" in prior_params:
            low = prior_params["low"]
        else:
            raise ValueError("`low` could not be inferred from prior")
    if high is None:
        if "high" in prior_params:
            high = prior_params["high"]
        else:
            raise ValueError("`high` could not be inferred from prior")

    dim_parameters = task.dim_parameters
    assert len(low) == dim_parameters
    assert len(high) == dim_parameters

    # Apply eps to bounds to avoid NaN evaluations
    low += eps
    high -= eps

    # Construct grid
    grid = torch.stack(
        torch.meshgrid(
            [torch.linspace(low[d], high[d], resolution) for d in range(dim_parameters)]
        )
    )  # dim_parameters x resolution x ... x resolution
    grid_flat = grid.view(
        dim_parameters, -1
    ).T  # resolution^dim_parameters x dim_parameters

    # Get log probability function (unnormalized log posterior)
    log_prob_fn = task._get_log_prob_fn(
        num_observation=num_observation,
        observation=observation,
        implementation="experimental",
        posterior=True,
        **kwargs,
    )

    total_evaluations = grid_flat.shape[0]
    log.info(f"Total evaluations: {total_evaluations}")

    batch_size = min(batch_size, total_evaluations)
    num_batches = int(total_evaluations / batch_size)

    log_probs = torch.empty([resolution for _ in range(dim_parameters)])
    for i in tqdm(range(num_batches)):
        ix_from = i * batch_size
        ix_to = ix_from + batch_size
        if ix_to > total_evaluations:
            ix_to = total_evaluations
        log_probs.view(-1)[ix_from:ix_to] = log_prob_fn(grid_flat[ix_from:ix_to, :])

    if save:
        log.info("Saving grid and log probs")
        torch.save(grid, "grid.pt")
        torch.save(log_probs, "log_probs.pt")

    probs = torch.exp(log_probs.view(-1))
    indices = torch.arange(0, len(probs))
    idxs = choice(indices, num_samples, True, probs)
    samples = grid_flat[idxs, :]
    num_unique_samples = len(torch.unique(samples, dim=0))
    log.info(f"Unique samples: {num_unique_samples}")

    toc = time.time()
    log.info(f"Finished after {toc-tic:.3f} seconds")

    return samples
