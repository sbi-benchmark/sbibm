import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pickle
import torch
from sbi import inference as inference

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from sbibm.tasks.ddm.utils import run_mcmc


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    automatic_transforms_enabled: bool = True,
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 100,
        "init_strategy": "sir",
        "sir_batch_size": 100,
        "sir_num_batches": 1000,
    },
    base_distribution: str = "lognormal",
    num_transforms: int = 5,
    num_bins: int = 5,
    tails: str = "rectified",
    tail_bound: float = 5.0,
    l_lower_bound: float = 1e-7,
    use_log_rts: bool = True,
    tail_bound_eps: float = 1e-5,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs Mixed Model for DDM.

    Args:
        task: Task instance
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of maf / mdn / made / nsf
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior_transformed = wrap_prior_dist(prior, transforms)
        simulator_transformed = wrap_simulator_fn(simulator, transforms)

    # Train choice net
    log.info(f"Running Mixed Model NLE")

    # Load model and given args.
    desired_args = [
        use_log_rts,
        num_bins,
        num_transforms,
        base_distribution,
        tails,
        tail_bound,
    ]
    if num_simulations == 100000:
        model_filename_base = "mm"
    elif num_simulations == 10000:
        model_filename_base = "mm10k"
        desired_args.append(tail_bound_eps)
    else:
        ValueError(f"No pretrained model with {num_simulations} budget.")

    with open(
        f"/home/janfb/qode/results/benchmarking_sbi/gridsearch{num_simulations}.p", "rb"
    ) as fh:
        _, a1 = pickle.load(fh).values()
        args = np.array(a1)

    # Search for desired args in pretrained models.
    try:
        model_idx = np.where((args == desired_args).all(1))[0][0]
    except IndexError:
        raise IndexError(f"Model with {desired_args} was not found.")

    with open(
        f"/home/janfb/qode/results/benchmarking_sbi/models/{model_filename_base}{model_idx}.p",
        "rb",
    ) as fh:
        mixed_model = pickle.load(fh)
    log.info(f"Loaded pretrained model with index {model_idx}.")

    potential_fn_mm = mixed_model.get_potential_fn(
        observation.reshape(-1, 1),
        transforms,
        prior_transformed,
        ll_lower_bound=np.log(l_lower_bound),
    )

    # Run MCMC in transformed space.
    log.info(f"Starting MCMC")
    samples = run_mcmc(
        prior=prior_transformed,
        potential_fn=potential_fn_mm,
        mcmc_parameters=mcmc_parameters,
        num_samples=num_samples,
    )

    # Return untransformed samples.
    return transforms.inv(samples), 100000, None
