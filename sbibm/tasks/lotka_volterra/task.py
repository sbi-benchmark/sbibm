from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, List, Optional

import pyro
import torch
from diffeqtorch import DiffEq
from pyro import distributions as pdist

import sbibm  # noqa -- needed for setting sysimage path
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.decorators import lazy_property


class LotkaVolterra(Task):
    def __init__(
        self,
        days: float = 20.0,
        saveat: float = 0.1,
        summary: Optional[str] = "subsample",
    ):
        """Lotka-Volterra model

        Args:
            N: Total population
            I0: Initial number of infected individuals
            R0: Initial number of recovered individuals
            days: Number of days
            saveat: When to save during solving
            summary: Summaries to use

        References:
            [1]: https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html
        """
        self.dim_data_raw = int(2 * (days / saveat + 1))

        if summary is None:
            dim_data = self.dim_data_raw
        elif summary == "subsample":
            dim_data = 20
        else:
            raise NotImplementedError
        self.summary = summary

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000020,  # observation 1
            1000030,  # observation 2
            1000034,  # observation 3
            1000013,  # observation 4
            1000004,  # observation 5
            1000011,  # observation 6
            1000012,  # observation 7
            1000039,  # observation 8
            1000041,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=4,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display="Lotka-Volterra",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        # Prior
        mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        self.prior_params = {
            "loc": torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2]),
            "scale": torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p]),
        }
        self.prior_dist = pdist.LogNormal(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.u0 = torch.tensor([30.0, 1.0])
        self.tspan = torch.tensor([0.0, days])
        self.days = days
        self.saveat = saveat

        # NOTE: For subsample statistic
        self.total_count = 1000  # TODO: Value?

    @lazy_property
    def de(self):
        return DiffEq(
            f=f"""
            function f(du,u,p,t)
                x, y = u
                alpha, beta, gamma, delta = p
                du[1] = alpha * x - beta * x * y
                du[2] = -gamma * y + delta * x * y
            end
            """,
            saveat=self.saveat,
            debug=False,  # 5
        )

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"]

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(
        self,
        max_calls: Optional[int] = None,
    ) -> Simulator:
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

            us = []
            for num_sample in range(num_samples):
                u, t = self.de(self.u0, self.tspan, parameters[num_sample, :])

                if u.shape != torch.Size([2, int(self.dim_data_raw / 2)]):
                    u = float("nan") * torch.ones((2, int(self.dim_data_raw / 2)))
                    u = u.double()

                if num_sample % 100 == 0:
                    gc.collect()
                    self.de.jl.eval("Base.GC.gc()")

                us.append(u.reshape(1, 2, -1))
            us = torch.cat(us).float()  # num_parameters x 2 x (days/saveat + 1)

            idx_contains_nan = torch.where(
                torch.isnan(us.reshape(num_samples, -1)).any(axis=1)
            )[
                0
            ]  # noqa
            idx_contains_no_nan = torch.where(
                ~torch.isnan(us.reshape(num_samples, -1)).any(axis=1)
            )[
                0
            ]  # noqa

            if self.summary is None:
                return us

            elif self.summary == "subsample":
                data = float("nan") * torch.ones((num_samples, self.dim_data))
                if len(idx_contains_nan) == num_samples:
                    return data

                us = us[:, :, ::21].reshape(num_samples, -1)
                data[idx_contains_no_nan, :] = pyro.sample(
                    "data",
                    pdist.LogNormal(
                        loc=torch.log(us[idx_contains_no_nan, :].clamp(1e-10, 10000.0)),
                        scale=0.1,
                    ).to_event(1),
                )
                return data

            else:
                raise NotImplementedError

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations"""
        if self.summary is None:
            return data.reshape(-1, 2, int(self.dim_data / 2))
        else:
            return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_observation: Observation number
            num_samples: Number of samples to generate
            observation: Observed data, if None, will be loaded using `num_observation`
            kwargs: Passed to run_mcmc

        Returns:
            Samples from reference posterior
        """
        from sbibm.algorithms.pyro.mcmc import run as run_mcmc
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        if num_observation is not None:
            initial_params = self.get_true_parameters(num_observation=num_observation)
        else:
            initial_params = None

        proposal_samples = run_mcmc(
            task=self,
            kernel="Slice",
            jit_compile=False,
            num_warmup=10_000,
            num_chains=1,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            initial_params=initial_params,
            automatic_transforms_enabled=True,
        )

        proposal_dist = get_proposal(
            task=self,
            samples=proposal_samples,
            prior_weight=0.1,
            bounded=True,
            density_estimator="flow",
            flow_model="nsf",
        )

        samples = run_rejection(
            task=self,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            batch_size=10_000,
            num_batches_without_new_max=1_000,
            multiplier_M=1.2,
            proposal_dist=proposal_dist,
        )

        return samples


if __name__ == "__main__":
    task = LotkaVolterra()
    task._setup(n_jobs=-1)
