from typing import Optional

import numpy as np
import pyabcranger
import torch
from sbi.simulators.simutils import simulate_in_batches
from tqdm import tqdm

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.torch import sample_with_weights

from .abcranger_utils import estimparam_args


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    batch_size: int = 1_000,
    ntrees: int = 500,
    nthreads: int = 1,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Random forest ABC using the abcranger toolbox.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        ntrees: Number of random forest trees
        nthreads: Number of cores for random forest regression

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = sbibm.get_logger(__name__)
    log.info(f"Starting to run RF-ABC")

    prior = task.get_prior()
    simulator = task.get_simulator()
    if observation is None:
        observation = task.get_observation(num_observation)

    # Simulate training data set
    log.info(f"Generating data set as reference table")
    thetas = prior(num_samples=num_simulations)

    xs = simulate_in_batches(
        simulator,
        theta=thetas,
        sim_batch_size=batch_size,
        num_workers=1,
        show_progress_bars=True,
    )

    assert not thetas.isnan().any()
    assert not xs.isnan().any()
    assert not observation.isnan().any()

    dim_thetas = thetas.shape[1]
    dim_xs = xs.shape[1]

    names_thetas = [f"t{i}" for i in range(dim_thetas)]
    names_xs = [f"x{i}" for i in range(dim_xs)]

    np_thetas = thetas.numpy().astype(np.float64)
    np_xs = xs.numpy().astype(np.float64)
    np_xo = observation.reshape(-1).numpy().astype(np.float64)

    # Put data in reference table
    reftable = pyabcranger.reftable(
        num_simulations,
        # nrecscen, sth with number of scenarios, would stick with 0.
        [0],
        [dim_thetas],  # Number of parameters per model, thus the list.
        names_thetas,  # param names
        names_xs,  # data dim names
        np_xs,  # data
        np_thetas,  # params
        # called scenarios internally, maybe the indices for the models.
        np.ones(num_simulations),
    )

    # Inference per dimension
    log.info(f"Running RF inference for each parameter separately")
    postres = [
        pyabcranger.estimparam(
            reftable,  # data
            np_xo,  # observation
            estimparam_args(
                dim, num_simulations, ntrees, nthreads
            ),  # options for cpp routines as string
            True,  # verbose flag (?)
            False,  # flag for weights: whether to return the weights for oob samples (?) or not.
        )
        for dim in tqdm(range(dim_thetas))
    ]

    # Get weights per dimension
    log.info(f"Generating independent posterior samples per parameter")
    samples = []
    for dim in range(dim_thetas):
        samples_dim = np.asanyarray(postres[dim].values_weights)[:, 0]
        weights_dim = np.asanyarray(postres[dim].values_weights)[:, 1]

        samples.append(
            sample_with_weights(
                values=torch.from_numpy(samples_dim),
                weights=torch.from_numpy(weights_dim),
                num_samples=num_samples,
            )
        )

    return torch.stack(samples, dim=1), simulator.num_simulations, None
