import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pickle
import torch
from sbi import inference as inference

from sbibm.algorithms.sbi.utils import (
    wrap_prior_dist,
)
from sbibm.tasks.task import Task
from sbibm.tasks.ddm.utils import run_mcmc


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    base_distribution: str = "lognormal",
    num_transforms: int = 5,
    num_bins: int = 5,
    tails: str = "rectified",
    tail_bound: float = 5.0,
    tail_bound_eps: float = 1e-5,
    automatic_transforms_enabled: bool = True,
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 100,
        "init_strategy": "sir",
        "sir_batch_size": 100,
        "sir_num_batches": 1000,
    },
    l_lower_bound: float = 1e-7,
    use_log_rts: bool = True,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs NLE for DDM using a mixed model for discrete choices and continuous rts.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        base_distribution: base distribution of neural spline flow, "normal" or "lognormal".
        num_transforms: number of transforms in neural spline flow.
        num_bins: number of bins for splines.
        tails: kind of spline tails, "linear" or "rectified"
        tail_bound: symmetric extend of the splines.
        tail_bound_eps: epsilon to add to tail bounds when rectified.
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        l_lower_bound: lower bound on single trial likelihood evaluations.
        use_log_rts: whether to transform rts to log space.
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    assert (
        task.name == "ddm"
    ), "This algorithm works only for the DDM task as it uses its analytical likeklihood."

    log = logging.getLogger(__name__)

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior_transformed = wrap_prior_dist(prior, transforms)

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
    # Load 100k or 10k model.
    if num_simulations == 100000:
        model_filename_base = "mm"
    elif num_simulations == 10000:
        model_filename_base = "mm10k"
        desired_args.append(tail_bound_eps)
    else:
        ValueError(f"No pretrained model with {num_simulations} budget.")

    path_to_arglist = (
        f"{Path(__file__).parent.resolve()}/nle_pretrained/arglist{num_simulations}.p"
    )

    with open(path_to_arglist, "rb") as fh:
        _, args = pickle.load(fh).values()
        args = np.array(args)

    # Search for desired args in pretrained models.
    try:
        model_idx = np.where((args == desired_args).all(1))[0][0]
    except IndexError:
        raise IndexError(f"Model with {desired_args} was not found.")

    path_to_model = f"{Path(__file__).parent.resolve()}/nle_pretrained/{model_filename_base}{model_idx}.p"

    with open(path_to_model, "rb") as fh:
        mixed_model = pickle.load(fh)
    log.info(f"Loaded pretrained model with index {model_idx}.")

    # Get potential function for mixed model.
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
    return transforms.inv(samples), num_simulations, None
