import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class GaussianLinearUniform(Task):
    def __init__(
        self, dim: int = 10, prior_bound: float = 1.0, simulator_scale: float = 0.1
    ):
        """Gaussian Linear Uniform

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
            simulator_scale: Standard deviation of noise in simulator.
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear Uniform",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "precision_matrix": torch.inverse(
                simulator_scale * torch.eye(self.dim_parameters),
            )
        }

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            return pyro.sample(
                "data",
                pdist.MultivariateNormal(
                    loc=parameters,
                    precision_matrix=self.simulator_params["precision_matrix"],
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses closed form solution with rejection sampling

        Args:
            num_samples: Number of samples to generate
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Samples from reference posterior
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        log = logging.getLogger(__name__)

        reference_posterior_samples = []

        sampling_dist = pdist.MultivariateNormal(
            loc=observation,
            precision_matrix=self.simulator_params["precision_matrix"],
        )

        # Reject samples outside of prior bounds
        counter = 0
        while len(reference_posterior_samples) < num_samples:
            counter += 1
            sample = sampling_dist.sample()
            if not torch.isinf(self.prior_dist.log_prob(sample).sum()):
                reference_posterior_samples.append(sample)

        reference_posterior_samples = torch.cat(reference_posterior_samples)
        acceptance_rate = float(num_samples / counter)

        log.info(
            f"Acceptance rate for observation {num_observation}: {acceptance_rate}"
        )

        return reference_posterior_samples


if __name__ == "__main__":
    task = GaussianLinearUniform()
    task._setup()
