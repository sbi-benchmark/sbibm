import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pickle
import torch
from sbi import inference as inference

from sbibm.algorithms.sbi.utils import wrap_prior_dist
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
        "num_chains": 10,
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
    else:
        prior_transformed = prior

    # Train choice net
    log.info(f"Running Mixed Model NLE")

    pretrained_model_filename = "mm_315_1"
    path_to_model = f"/home/janfb/qode/sbibm/sbibm/algorithms/lan/nle_pretrained/{pretrained_model_filename}.p"

    with open(path_to_model, "rb") as fh:
        mixed_model = pickle.load(fh)
    log.info(f"Loaded pretrained model {pretrained_model_filename}.")

    # Get potential function for mixed model.
    potential_fn_mm = mixed_model.get_potential_fn(
        observation.reshape(-1, 1),
        transforms,
        # Pass untransformed prior and correct internally with ladj.
        prior=prior,
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
