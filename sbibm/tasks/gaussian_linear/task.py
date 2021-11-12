from pathlib import Path
from typing import Callable, Optional

import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class GaussianLinear(Task):
    def __init__(
        self, dim: int = 10, prior_scale: float = 0.1, simulator_scale: float = 0.1
    ):
        """Gaussian Linear

        Inference of mean under Gaussian prior in 2D

        Args:
            dim: Dimensionality of parameters and data
            prior_scale: Standard deviation of prior
            simulator_scale: Standard deviation of noise in simulator
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "loc": torch.zeros((self.dim_parameters,)),
            "precision_matrix": torch.inverse(
                prior_scale * torch.eye(self.dim_parameters)
            ),
        }

        self.prior_dist = pdist.MultivariateNormal(**self.prior_params)
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

    def _get_reference_posterior(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> pdist.Distribution:
        """Gets posterior

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Posterior distribution
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        N = 1
        covariance_matrix = torch.inverse(
            self.prior_params["precision_matrix"]
            + N * self.simulator_params["precision_matrix"]
        )
        loc = torch.matmul(
            covariance_matrix,
            (
                N
                * torch.matmul(
                    self.simulator_params["precision_matrix"], observation.reshape(-1)
                )
                + torch.matmul(
                    self.prior_params["precision_matrix"],
                    self.prior_params["loc"],
                )
            ),
        )

        posterior = pdist.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses closed form solution

        Args:
            num_samples: Number of samples to generate
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Samples from reference posterior
        """
        posterior = self._get_reference_posterior(
            num_observation=num_observation,
            observation=observation,
        )

        return posterior.sample((num_samples,))


if __name__ == "__main__":
    task = GaussianLinear()
    task._setup()
