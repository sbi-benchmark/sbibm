import logging
import pathlib
from typing import Any, Dict, Optional, Tuple

import keras
import torch
from sbibm.tasks.task import Task
from sbibm.tasks.ddm.utils import LANPotentialFunctionProvider, run_mcmc

from sbibm.algorithms.sbi.utils import wrap_prior_dist


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
    """Runs pretrained LAN based on analytical likelihood targets.

    Args:
        task: Task instance
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
    log.info(f"Running LAN pretrained with analytical targets.")
    # Set LAN budget from paper.
    lan_budget = int(1e5 * 1.5e6)

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    # Maybe transform to unconstrained parameter space for MCMC.
    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior_transformed = wrap_prior_dist(prior, transforms)
    else:
        prior_transformed = prior

    num_trials = observation.shape[1]
    # sbi needs the trials in first dimension.
    observation_sbi = observation.reshape(num_trials, 1)

    # network trained on analytical likelihood for 4-param ddm
    lan_ana_path = f"{pathlib.Path(__file__).parent.resolve()}/lan_pretrained/model_final_ddm_analytic.h5"
    # load weights as keras model
    lan_ana = keras.models.load_model(lan_ana_path, compile=False)

    # Use potential function provided refactored from SBI toolbox for LAN.
    potential_fn_lan = LANPotentialFunctionProvider(transforms, lan_ana, l_lower_bound)

    # Run MCMC in transformed space.
    samples = run_mcmc(
        prior=prior_transformed,
        # Pass original prior to pf and correct potential with ladj.
        potential_fn=potential_fn_lan(
            prior=prior,
            sbi_net=None,
            x=observation_sbi,
            mcmc_method=mcmc_method,
        ),
        mcmc_parameters=mcmc_parameters,
        num_samples=num_samples,
    )

    # Return untransformed samples.
    return transforms.inv(samples), lan_budget, None
