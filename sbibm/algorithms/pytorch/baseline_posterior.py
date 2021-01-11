from typing import Any

import numpy as np
import torch

import sbibm
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_observation: int,
    rerun: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Random samples from saved reference posterior as baseline

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        rerun: Whether to rerun reference or load from disk

    Returns:
        Random samples from reference posterior
    """
    log = sbibm.get_logger(__name__)

    if "num_simulations" in kwargs:
        log.warn(
            "`num_simulations` was passed as a keyword but will be ignored, since this is a baseline method."
        )

    if rerun:
        return task._sample_reference_posterior(
            num_samples=num_samples, num_observation=num_observation
        )
    else:
        reference_posterior_samples = task.get_reference_posterior_samples(
            num_observation
        )

        return reference_posterior_samples[
            np.random.randint(reference_posterior_samples.shape[0], size=num_samples),
            :,
        ]
