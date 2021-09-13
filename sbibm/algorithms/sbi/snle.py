import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn
from sbi.inference.posteriors.likelihood_based_posterior import (
    PotentialFunctionProvider,
)

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from sbibm.algorithms.lan.utils import run_mcmc


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    automatic_transforms_enabled: bool = True,
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 100,
        "init_strategy": "sir",
        "sir_batch_size": 100,
        "sir_num_batches": 1000,
    },
    z_score_x: bool = True,
    z_score_theta: bool = True,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    num_transforms: int = 1,
    num_bins: int = 10,
    l_lower_bound: float = 1e-7,
    use_pretrained: bool = False,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NLE from `sbi`

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

    if num_rounds == 1:
        log.info(f"Running NLE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNLE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior_transformed = wrap_prior_dist(prior, transforms)
        simulator_transformed = wrap_simulator_fn(simulator, transforms)

    # Load pretrained network.
    density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
        num_transforms=num_transforms,
        num_bins=num_bins,
    )
    inference_method = inference.SNLE_A(
        density_estimator=density_estimator_fun,
        prior=prior,
    )

    posteriors = []
    proposal = prior
    num_trials = observation.shape[1]
    # sbi needs the trials in first dimension.
    observation_sbi = observation.reshape(num_trials, 1)

    for r in range(num_rounds):
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        density_estimator = inference_method.append_simulations(
            theta, x, from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
            show_train_summary=True,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
        )
        if r > 1:
            mcmc_parameters["init_strategy"] = "latest_sample"
        posterior = inference_method.build_posterior(
            density_estimator,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
        )
        # Copy hyperparameters, e.g., mcmc_init_samples for "latest_sample" strategy.
        if r > 0:
            posterior.copy_hyperparameters_from(posteriors[-1])
        proposal = posterior.set_default_x(observation_sbi)
        posteriors.append(posterior)

    potential_fn_snl = PotentialFunctionProvider()
    # Call to initialize.
    # Use transformed prior for MCMC.
    potential_fn_snl(
        prior_transformed,
        density_estimator,
        observation_sbi,
        mcmc_method,
        transforms,
        l_lower_bound,
    )

    # Run MCMC in transformed space.
    samples = run_mcmc(
        prior=prior_transformed,
        potential_fn=potential_fn_snl.posterior_potential,
        mcmc_parameters=mcmc_parameters,
        num_samples=num_samples,
    )

    # Return untransformed samples.
    return transforms.inv(samples), simulator.num_simulations, None
