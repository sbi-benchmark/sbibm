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
        max_num_epochs=10,
    )

    # network trained on KDE likelihood for 4-param ddm
    lan_kde_model = "/home/janfb/qode/sbibm/sbibm/algorithms/lan/model_final_ddm.h5"
    # load weights as keras model
    lan_kde = keras.models.load_model(lan_kde_model, compile=False)
    inference_method._x_shape = torch.Size([1, 1])

    posterior = inference_method.build_posterior(
        None, mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters
    )
    posterior = wrap_posterior(posterior, transforms)

    # Run MCMC like for reference posterior with LAN potential function.

    samples = posterior.sample(
        (num_samples,),
        # Monkey patch LAN likelihood into SBI potential function provider
        potential_fn_provider=PotentialFunctionProvider(transforms, lan_kde),
        x=observation_sbi,
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

    def __init__(self, transforms, lan_net) -> None:

        self.transforms = transforms
        self.lan_net = lan_net

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
        self.likelihood_nn = self.lan_net
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
        theta = torch.as_tensor(theta, dtype=torch.float32)

        # Notice opposite sign to pyro potential.
        return self.log_likelihood(theta).cpu() + self.prior.log_prob(theta)

    def _log_likelihoods_over_trials(self, x, theta_unconstrained):

        # move to parameters to constrained space.
        theta = self.transforms.inv(theta_unconstrained)

        theta_repeated, x_repeated = NeuralPosterior._match_theta_and_x_batch_shapes(
            theta=theta, x=atleast_2d(x)
        )
        assert (
            x_repeated.shape[0] == theta_repeated.shape[0]
        ), "x and theta must match in batch shape."

        # x has shape (num_trials*batch_size, 1)
        # theta is a batch of (batch_size*num_trials, 3)

        rts = abs(x_repeated)
        # Decode choices from sign of RT.
        cs = torch.ones(x_repeated.shape[0], 1)
        cs[x_repeated < 0] *= -1

        # maybe add ndt column to params.
        if theta.shape[1] < 4:
            theta_lan = torch.cat(
                (theta_repeated, torch.zeros(theta_repeated.shape[0], 1)), dim=1
            )
        else:
            theta_lan = theta_repeated

        # transform boundary separation into symmetric boundary.
        theta_lan[:, 1] *= 0.5

        # stack thetas, rts and choices for keras model.
        theta_x_stack = torch.cat((theta_lan, rts, cs), dim=1)

        log_likelihood_trial_batch = torch.tensor(
            self.likelihood_nn.predict_on_batch(theta_x_stack.numpy()),
            dtype=torch.float32,
        )

        # Reshape to (parameters by x-trials) and sum over trials.
        log_likelihood_trial_sum_constrained = log_likelihood_trial_batch.reshape(
            x.shape[0], -1
        ).sum(0)

        # move likelihood to unconstrained space.
        log_abs_det = self.transforms.log_abs_det_jacobian(theta, theta_unconstrained)
        # If no transforms are used torch returns parameter dimensions.
        if log_abs_det.ndim > 1:
            log_abs_det = log_abs_det.sum(-1)
        # trial log likelihood are in columns, parameters are the same for each row.
        # thus subtract log_abs_det_jacobian for each row:
        log_likelihood_trial_sum_unconstrained = (
            log_likelihood_trial_sum_constrained - log_abs_det
        )

        # sum over trial-log likelihoods in second dim.
        return log_likelihood_trial_sum_unconstrained
