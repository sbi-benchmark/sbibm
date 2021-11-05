import logging
import time
from math import ceil
from typing import Optional

import elfi
import numpy as np
import torch

from sbibm.algorithms.elfi.utils.prior import build_prior
from sbibm.algorithms.elfi.utils.wrapper import Simulator
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_chains: int = 10,
    num_warmup: int = 1000,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Runs BOLFI from elfi package

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_chains: Number of chains
        num_warmup: Warmup steps

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(__name__)
    log.warn("ELFI is not fully supported yet!")

    # Initialize model object
    m = elfi.ElfiModel()

    # Prior
    bounds = build_prior(task=task, model=m)

    # Observation
    if observation is None:
        observation = task.get_observation(num_observation)
    observation = observation.numpy()

    # Simulator
    simulator = task.get_simulator(max_calls=num_simulations)
    elfi.Simulator(
        Simulator(simulator),
        *[m[f"parameter_{dim}"] for dim in range(task.dim_parameters)],
        observed=observation,
        name=task.name,
    )

    # Euclidean distance
    elfi.Distance("euclidean", m[task.name], name="distance")

    # Log distance
    elfi.Operation(np.log, m["distance"], name="log_distance")

    # Inference
    num_samples_per_chain = ceil(num_samples / num_chains)
    tic = time.time()
    bolfi = elfi.BOLFI(model=m, target_name="log_distance", bounds=bounds)
    bolfi.fit(n_evidence=num_simulations)
    result_BOLFI = bolfi.sample(
        num_samples_per_chain + num_warmup,
        warmup=num_warmup,
        n_chains=num_chains,
        info_freq=int(100),
    )
    toc = time.time()

    samples = torch.from_numpy(result_BOLFI.samples_array.astype(np.float32)).reshape(
        -1, task.dim_parameters
    )[:num_samples, :]

    assert samples.shape[0] == num_samples

    # TODO: return log prob of true parameters

    return samples, simulator.num_simulations, None
