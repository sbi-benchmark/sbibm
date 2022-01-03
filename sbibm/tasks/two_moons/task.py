import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pyro
import torch
from pyro import distributions as pdist

import sbibm
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.pyro import make_log_prob_grad_fn


class TwoMoons(Task):
    def __init__(self):
        """Two Moons"""

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000011,  # observation 1
            1000001,  # observation 2
            1000002,  # observation 3
            1000003,  # observation 4
            1000013,  # observation 5
            1000005,  # observation 6
            1000006,  # observation 7
            1000007,  # observation 8
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=2,
            name=Path(__file__).parent.name,
            name_display="Two Moons",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        prior_bound = 1.0
        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "a_low": -math.pi / 2.0,
            "a_high": +math.pi / 2.0,
            "base_offset": 0.25,
            "r_loc": 0.1,
            "r_scale": 0.01,
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
            num_samples = parameters.shape[0]

            a_dist = (
                pdist.Uniform(
                    low=self.simulator_params["a_low"],
                    high=self.simulator_params["a_high"],
                )
                .expand_by((num_samples, 1))
                .to_event(1)
            )
            a = a_dist.sample()

            r_dist = (
                pdist.Normal(
                    self.simulator_params["r_loc"], self.simulator_params["r_scale"]
                )
                .expand_by((num_samples, 1))
                .to_event(1)
            )
            r = r_dist.sample()

            p = torch.cat(
                (
                    torch.cos(a) * r + self.simulator_params["base_offset"],
                    torch.sin(a) * r,
                ),
                dim=1,
            )

            return self._map_fun(parameters, p)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    @staticmethod
    def _map_fun(parameters: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return p + torch.cat((-torch.abs(z0), z1), dim=1)

    @staticmethod
    def _map_fun_inv(parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return x - torch.cat((-torch.abs(z0), z1), dim=1)

    def _likelihood(
        self,
        parameters: torch.Tensor,
        data: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)

        assert parameters.shape[1] == self.dim_parameters
        assert data.shape[1] == self.dim_data

        p = self._map_fun_inv(parameters, data).squeeze(0)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        u = p[:, 0] - self.simulator_params["base_offset"]
        v = p[:, 1]

        r = torch.sqrt(u ** 2 + v ** 2)
        L = -0.5 * (
            (r - self.simulator_params["r_loc"]) / self.simulator_params["r_scale"]
        ) ** 2 - 0.5 * torch.log(
            2 * torch.tensor([math.pi]) * self.simulator_params["r_scale"] ** 2
        )

        if len(torch.where(u < 0.0)[0]) > 0:
            L[torch.where(u < 0.0)[0]] = -torch.tensor(math.inf)

        return L if log else torch.exp(L)

    def _get_transforms(
        self,
        *args,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"parameters": torch.distributions.transforms.IndependentTransform(torch.distributions.transforms.identity_transform, 1) }

    def _get_log_prob_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Callable:
        """Get potential function and initial parameters

        The potential function returns the unnormalized negative log
        posterior probability, and is useful to establish and verify
        the reference posterior.

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Potential function and proposal for initial parameters, e.g., to start MCMC
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        prior_dist = self.get_prior_dist()

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        observation = self.unflatten_data(observation)

        def log_prob_fn(parameters):
            if type(parameters) == dict:
                parameters = parameters["parameters"]
            return self._likelihood(
                parameters=parameters, data=observation, log=True
            ) + prior_dist.log_prob(parameters)

        return log_prob_fn

    def _get_log_prob_grad_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Callable:
        lpgf = make_log_prob_grad_fn(
            self._get_log_prob_fn(
                num_observation=num_observation, observation=observation, **kwargs
            )
        )

        def log_prob_grad_fn(parameters):
            num_params = parameters.shape[0]
            grads = []
            for i in range(num_params):
                _, grad = lpgf({"parameters": parameters[i]})
                grads.append(grad)
            if len(grads) > 1:
                return torch.cat(grads).reshape(1, -1)
            else:
                return grad

        return log_prob_grad_fn

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: int,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses closed form solution

        Args:
            num_samples: Number of samples to generate
            num_observation: Observation number
            observation: Observed data, if None, will be loaded using `num_observation`

        Returns:
            Samples from reference posterior
        """
        log = sbibm.get_logger(__name__)

        if observation is None:
            observation = self.get_observation(num_observation)

        reference_posterior_samples = []

        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(-ang)
        s = torch.sin(-ang)

        simulator = self.get_simulator()

        reference_posterior_samples = []
        counter = 0
        while len(reference_posterior_samples) < num_samples:
            counter += 1

            p = simulator(torch.zeros(1, 2))
            q = torch.zeros(2)
            q[0] = p[0, 0] - observation[0, 0]
            q[1] = observation[0, 1] - p[0, 1]

            if np.random.rand() < 0.5:
                q[0] = -q[0]

            sample = torch.tensor([[c * q[0] - s * q[1], s * q[0] + c * q[1]]])

            is_outside_prior = torch.isinf(self.prior_dist.log_prob(sample).sum())

            if len(reference_posterior_samples) > 0:
                is_duplicate = sample in torch.cat(reference_posterior_samples)
            else:
                is_duplicate = False

            if not is_outside_prior and not is_duplicate:
                reference_posterior_samples.append(sample)

        reference_posterior_samples = torch.cat(reference_posterior_samples)
        acceptance_rate = float(num_samples / counter)

        log.info(
            f"Acceptance rate for observation {num_observation}: {acceptance_rate}"
        )

        return reference_posterior_samples


if __name__ == "__main__":
    task = TwoMoons()
    task._setup()
