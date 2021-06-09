import logging
from typing import Any, Dict, Optional, Tuple, Callable

import keras
import numpy as np
import torch
from sbibm.tasks.task import Task
from sbibm.tasks.ddm.utils import PotentialFunctionProvider

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
import sbi.inference as inference


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
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs LANs using pretrained nets.

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

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)
    lan_budget = int(1e5 * 1.5e6)

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    num_trials = observation.shape[1]
    # sbi needs the trials in first dimension.
    observation_sbi = observation.reshape(num_trials, 1)

    # Define dummy sbi object and plug in LAN potential function.
    inference_method = inference.SNLE_A(
        density_estimator="nsf",
        prior=prior,
        device="cpu",
    )
    theta, x = inference.simulate_for_sbi(
        simulator,
        prior,
        num_simulations=1000,
        simulation_batch_size=100,
    )

    inference_method.append_simulations(theta, x, from_round=0).train(
        training_batch_size=100,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        max_num_epochs=2,
    )

    # network trained on KDE likelihood for 4-param ddm
    lan_ana_model = (
        "/home/janfb/qode/sbibm/sbibm/algorithms/lan/model_final_ddm_analytic.h5"
    )
    # load weights as keras model
    lan_ana = keras.models.load_model(lan_ana_model, compile=False)
    inference_method._x_shape = torch.Size([1, 1])

    posterior = inference_method.build_posterior(
        None, mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters
    )
    posterior = wrap_posterior(posterior, transforms)

    # Run MCMC like for reference posterior with LAN potential function.

    samples = posterior.sample(
        (num_samples,),
        # Monkey patch LAN likelihood into SBI potential function provider
        **dict(
            potential_fn_provider=PotentialFunctionProvider(transforms, lan_ana),
            x=observation_sbi,
        ),
    ).detach()

    return samples, lan_budget, None
