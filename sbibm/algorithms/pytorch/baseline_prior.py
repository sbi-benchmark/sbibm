from typing import Any

import torch

import sbibm
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Random samples from prior as baseline

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior

    Returns:
        Random samples from prior
    """
    log = sbibm.get_logger(__name__)

    if "num_simulations" in kwargs:
        log.warn(
            "`num_simulations` was passed as a keyword but will be ignored, since this is a baseline method."
        )

    prior = task.get_prior()
    return prior(num_samples=num_samples)
