import logging
from typing import Any, Dict, Optional, Tuple, Callable

import keras
import numpy as np
import torch
from torch import Tensor, nn
from sbibm.tasks.task import Task

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
import sbi.inference as inference

from sbi.utils.torchutils import ScalarFloat, atleast_2d, ensure_theta_batched
from sbi.utils import within_support
from sbi.inference.posteriors.base_posterior import NeuralPosterior


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
        max_num_epochs=5,
    )

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
            potential_fn_provider=PotentialFunctionProvider(
                transforms, task, l_lower_bound=1e-7
            ),
            x=observation_sbi,
        ),
    ).detach()

    return samples, num_simulations, None


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __init__(self, transforms, task, l_lower_bound) -> None:

        self.transforms = transforms
        self.task = task
        self.l_lower_bound = l_lower_bound

    def __call__(
        self, prior, sbi_net: nn.Module, x: Tensor, mcmc_method: str
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$. Can be a batch of iid
                x.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.prior = prior
        self.device = "cpu"
        self.x = atleast_2d(x).to(self.device)
        return self.np_potential

    def log_likelihood(self, theta: Tensor, track_gradients: bool = False) -> Tensor:
        """Return log likelihood of fixed data given a batch of parameters."""

        log_likelihoods = self._log_likelihoods_over_trials(
            self.x,
            ensure_theta_batched(theta).to(self.device),
        )

        return log_likelihoods

    def np_potential(self, theta: np.array) -> ScalarFloat:
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        parameters = torch.as_tensor(theta, dtype=torch.float32)

        # We need to calculate likelihoods in constrained space.
        parameters_constrained = self.transforms.inv(parameters)

        # Get likelihoods from DiffModels.jl in constrained space.
        log_likelihood_constrained = self.task.get_log_likelihood(
            parameters_constrained, self.x.reshape(1, -1), self.l_lower_bound
        )
        # But we need log probs in unconstrained space. Get log abs det jac
        log_abs_det = self.transforms.log_abs_det_jacobian(
            parameters_constrained, parameters
        )
        # Without transforms, logabsdet returns second dimension.
        if log_abs_det.ndim > 1:
            log_abs_det = log_abs_det.sum(-1)

        # Likelihood in unconstrained space is:
        # prob_constrained * 1/abs_det_jacobian
        # log_prob_constrained - log_abs_det
        log_likelihood = log_likelihood_constrained - log_abs_det
        return log_likelihood + self.prior.log_prob(parameters)

    def _log_likelihoods_over_trials(self, x, theta_unconstrained):

        # move to parameters to constrained space.
        theta = self.transforms.inv(theta_unconstrained)

        # Get likelihoods from DiffModels.jl in constrained space.
        log_likelihood_constrained = self.task.get_log_likelihood(
            theta, x.reshape(1, -1), self.l_lower_bound
        )
        # But we need log probs in unconstrained space. Get log abs det jac
        log_abs_det = self.transforms.log_abs_det_jacobian(theta, theta_unconstrained)
        # Without transforms, logabsdet returns second dimension.
        if log_abs_det.ndim > 1:
            log_abs_det = log_abs_det.sum(-1)
            assert log_abs_det.numel() == log_likelihood_constrained.numel()

        # Likelihood in unconstrained space is:
        # prob_constrained * 1/abs_det_jacobian
        # log_prob_constrained - log_abs_det
        log_likelihood = log_likelihood_constrained - log_abs_det

        return log_likelihood
