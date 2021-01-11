import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from sbi import inference as inference
from sbi.inference.posteriors.likelihood_based_posterior import LikelihoodBasedPosterior
from torch import Tensor

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task


class SynthLikNet(nn.Module):
    def __init__(self, simulator, num_simulations_per_step=100, diag_eps=0.0):
        self.simulator = simulator
        self.num_simulations_per_step = num_simulations_per_step
        self.diag_eps = diag_eps

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def log_prob(self, inputs: Tensor, context=Optional[Tensor]) -> Tensor:
        thetas = context
        observation = inputs

        log_probs = []
        for i in range(thetas.shape[0]):
            xs = self.simulator(
                thetas[i, :].reshape(1, -1).repeat(self.num_simulations_per_step, 1)
            )

            # Estimate mean and covariance of MVN
            m = torch.mean(xs, dim=0)
            xm = xs - m
            S = torch.matmul(xm.T, xm) / xs.shape[0]
            S = S + self.diag_eps * torch.eye(xs.shape[1])

            # Score
            dist = torch.distributions.MultivariateNormal(loc=m, covariance_matrix=S)
            log_probs.append(dist.log_prob(observation[i, :].reshape(1, -1)))

        return torch.cat(log_probs)

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_simulations_per_step: int = 100,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    automatic_transforms_enabled: bool = False,
    mcmc_method: str = "slice_np",
    mcmc_parameters: Dict[str, Any] = {},
    diag_eps: float = 0.0,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Runs (S)NLE from `sbi`

    Args:
        task: Task instance
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_simulations_per_step: Number of simulations per MCMC step
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        diag_eps: Epsilon applied to diagonal

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    log.info(f"Running SL")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator()

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    prior = wrap_prior_dist(prior, transforms)
    simulator = wrap_simulator_fn(simulator, transforms)

    likelihood_estimator = SynthLikNet(
        simulator=simulator,
        num_simulations_per_step=num_simulations_per_step,
        diag_eps=diag_eps,
    )

    posterior = LikelihoodBasedPosterior(
        method_family="snle",
        neural_net=likelihood_estimator,
        prior=prior,
        x_shape=observation.shape,
        mcmc_parameters=mcmc_parameters,
    )

    posterior.set_default_x(observation)

    posterior = wrap_posterior(posterior, transforms)

    # assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, None
