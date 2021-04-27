from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, List, Optional

import pyro
import torch
from julia import Julia
from pyro import distributions as pdist
from utils import DDMJulia

import sbibm  # noqa -- needed for setting sysimage path
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.decorators import lazy_property


class DDM(Task):
    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
    ):
        """Drift-diffusion model.

        Args:
            dt: integration step size in s.
        """
        self.dt = dt
        self.num_trials = num_trials

        super().__init__(
            dim_parameters=2,
            dim_data=2 * num_trials,
            name=Path(__file__).parent.name,
            name_display="DDM",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=torch.arange(10),
        )

        # Prior
        self.prior_params = {
            "low": torch.tensor([-2, 0.5]),
            "high": torch.tensor([2, 2]),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

    @lazy_property
    def ddm(self):
        return DDMJulia(dt=self.dt, num_trials=self.num_trials)

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return ["v", "a"]

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
            rts, choices = self.ddm.simulate(
                parameters[:, 0].numpy(), parameters[:, 1].numpy()
            )
            return torch.cat(
                (
                    torch.tensor(rts, dtype=torch.float32),
                    torch.tensor(choices, dtype=torch.float32),
                ),
                dim=1,
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_likelihood(self, parameters, data):
        """Return likelihood given parameters and data.

        Takes product of likelihoods across iid trials.

        Batch dimension is only across parameters, the data is fixed.
        """
        num_samples = parameters.shape[0]
        num_trials = int(data.shape[1] / 2)

        likelihoods = torch.zeros(num_samples)
        for idx in range(num_samples):
            likelihoods[idx] = self.ddm.likelihood(
                float(parameters[idx, 0]),
                # Take abs because bound is symmetric.
                abs(float(parameters[idx, 1])),
                data[0, :num_trials].numpy(),
                data[0, num_trials:].numpy(),
            )

        return likelihoods

    def get_potential_function(self, data) -> Callable:
        """Return potential function for fixed data.

        Potential: $-[\log r(x_o, \theta) + \log p(\theta)]$

        The data can consists of multiple iid trials.
        Then the overall likelihood is defined as the product over iid likelihood.
        """
        # TODO: take into account inverse transform on parameters?
        def potential(parameters):
            prior_lobprobs = self.get_prior_dist().log_prob(parameters["parameters"])
            log_likelihoods = self.get_likelihood(parameters["parameters"], data).log()

            return -(log_likelihoods + prior_lobprobs)

        return potential

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

        num_chains = 1
        num_warmup = 10_000

        proposal_samples = run_mcmc(
            task=self,
            # Pass observation to fix data for potential function.
            potential_fn=self.get_potential_function(
                self.get_observation(num_observation)
            ),
            kernel="Slice",
            jit_compile=False,
            num_warmup=num_warmup,
            num_chains=num_chains,
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
    task = DDM()
    task._setup(n_jobs=-1)
