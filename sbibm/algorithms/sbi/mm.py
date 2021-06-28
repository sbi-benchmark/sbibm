import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from sbibm.tasks.ddm.utils import (
    BernoulliMN,
    MixedModelSyntheticDDM,
    run_mcmc,
    train_choice_net,
)


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    neural_net: str = "nsf",
    base_distribution: str = "normal",
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
        "num_init_workers": 5,
    },
    z_score_x: bool = True,
    z_score_theta: bool = True,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    num_transforms: int = 1,
    num_bins: int = 10,
    tail_bound: float = 3.0,
    tail_bound_eps: float = 1e-10,
    l_lower_bound: float = 1e-7,
    use_log_rts: bool = False,
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
    log.info(f"Training Bernoulli choice net")
    theta = prior.sample((num_simulations,))
    theta, x = inference.simulate_for_sbi(
        simulator,
        prior,
        num_simulations=num_simulations,
        simulation_batch_size=simulation_batch_size,
    )
    rts = abs(x)
    choices = torch.ones_like(x)
    choices[x < 0] = 0
    theta_and_choices = torch.cat((theta, choices), dim=1)
    choice_net, val_lp = train_choice_net(
        theta,
        choices,
        BernoulliMN(n_input=theta.shape[1], n_hidden_units=20, n_hidden_layers=2),
    )

    # Train rt flow
    log.info(f"Training RT flow")
    density_estimator_fun = likelihood_nn(
        model=neural_net,
        num_transforms=num_transforms,
        hidden_features=hidden_features,
        num_bins=num_bins,
        base_distribution=base_distribution,
        tail_bound=tail_bound,
        tail_bound_eps=tail_bound_eps,
        z_score_theta=z_score_theta,
        z_score_x=z_score_x,
    )
    inference_method = inference.SNLE(
        density_estimator=density_estimator_fun,
        prior=prior,
    )
    inference_method = inference_method.append_simulations(
        theta=theta_and_choices,
        x=torch.log(rts) if use_log_rts else rts,
        from_round=0,
    )
    rt_flow = inference_method.train(
        training_batch_size=training_batch_size,
        show_train_summary=False,
        stop_after_epochs=stop_after_epochs,
        validation_fraction=validation_fraction,
    )

    mixed_model = MixedModelSyntheticDDM(choice_net, rt_flow, use_log_rts=use_log_rts)
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
    return transforms.inv(samples), simulator.num_simulations, None
