import logging
from typing import Any, Dict, Optional, Tuple

import torch

from sbibm.tasks.task import Task

from sbibm.algorithms.sbi.utils import wrap_prior_dist
from sbibm.tasks.ddm.utils import run_mcmc


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    automatic_transforms_enabled: bool = True,
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 100,
        "init_strategy": "sir",
        "sir_batch_size": 1000,
        "sir_num_batches": 100,
    },
    l_lower_bound: float = 1e-7,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs MCMC with analytical DDM likelihood.

    Args:
        task: Task instance, here DDM.
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        l_lower_bound: lower bound for single trial likelihood evaluations.

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    assert (
        task.name == "ddm"
    ), "This algorithm works only for the DDM task as it uses its analytical likeklihood."

    log = logging.getLogger(__name__)
    log.info(f"Running MCMC with analytical likelihoods from Julia package.")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior_transformed = wrap_prior_dist(prior, transforms)

    # sbi needs the trials in first dimension.

    llj = task._get_log_prob_fn(
        None,
        observation,
        "experimental",
        posterior=True,
        automatic_transforms_enabled=automatic_transforms_enabled,
        l_lower_bound=l_lower_bound,
    )

    def potential_fn_julia(theta):
        theta = torch.as_tensor(theta, dtype=torch.float32)

        return llj(theta)

    # Run MCMC in transformed space.
    samples = run_mcmc(
        prior=prior_transformed,
        potential_fn=potential_fn_julia,
        mcmc_parameters=mcmc_parameters,
        num_samples=num_samples,
    )

    # Return untransformed samples.
    return transforms.inv(samples), num_simulations, None
