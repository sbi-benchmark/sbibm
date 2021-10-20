from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv, save_tensor_to_csv


class norefposterior(Task):

    def __init__(self):
        """Forward-only simulator (without a reference posterior)
        """

        self.max_axis = 100
        dim_data = 2*self.max_axis
        name_display = "norefposterior"

        # TODO: no clear what purpose these serve
        # Observation seeds to use when generating ground truth
        # Avoiding extremely spiked posteriors, e.g., 1000006, 1000007, ...
        observation_seeds = [
            1000000,  # observation 1
            1000001,  # observation 2
            1000002,  # observation 3
            1000003,  # observation 4
            1000004,  # observation 5
            1000005,  # observation 6
            1000010,  # observation 7
            1000012,  # observation 8
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=4,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display=name_display,
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        self.prior_params = {
            "low": torch.tensor([20,20,5,5]).float(),
            "high": torch.tensor([80,80,15,15]).float(),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

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

            m_ = torch.stack(
                (parameters[:, [0]].squeeze(), parameters[:, [1]].squeeze())
            ).T
            if m_.dim() == 1:
                m_.unsqueeze_(0)

            m = torch.broadcast_to(m_, (self.max_axis, self.max_axis, *m_.shape))

            s1 = parameters[:, [2]].squeeze() #** 2
            s2 = parameters[:, [3]].squeeze() #** 2

            # TODO: double check if the formulae below make sense
            # (mostly used as placeholders for now)
            S = torch.empty((self.max_axis, self.max_axis, num_samples, 2, 2))
            S[..., 0, 0] = s1 ** 2
            S[..., 0, 1] = s1 * s2
            S[..., 1, 0] = 0. #s1 * s2
            S[..., 1, 1] = s2 ** 2

            # Add eps to diagonal to ensure PSD
            eps = 0.000001
            S[..., 0, 0] += eps
            S[..., 1, 1] += eps

            assert S.shape == (self.max_axis, self.max_axis, num_samples, 2, 2), f"{name_display} :: cov matrix {S.shape} != expectation"
            assert m.shape == (self.max_axis, self.max_axis, num_samples, 2), f"{name_display} :: mean vector {m.shape} != expectation"

            data_dist = pdist.MultivariateNormal(
                m.float(),
                S.float()
            )

            # discretize on grid
            x = torch.linspace(0, self.max_axis, self.max_axis).detach()
            y = torch.linspace(0, self.max_axis, self.max_axis).detach()
            xx, yy = torch.meshgrid(x, y)

            # prepare grid values to eval MultivariateNormal upon
            val = torch.swapaxes(torch.stack((xx.flatten(), yy.flatten())),
                                 1,
                                 0).float()

            assert val.shape == (self.max_axis*self.max_axis,2), f"grid values shape {val.shape} unexpected"

            valr = val.reshape(self.max_axis, self.max_axis,
                               2)

            assert valr.shape == (self.max_axis, self.max_axis, 2), f"{name_display} :: batched grid values shape {valr.shape[1:]} unexpected"

            valb = torch.swapaxes(torch.broadcast_to(valr, (num_samples, *valr.shape)).detach(),
                                  0,
                                  2)
            assert valb.shape == (self.max_axis, self.max_axis, num_samples, 2), f"{name_display} :: batched grid values shape {valb.shape[1:]} unexpected"

            # TODO: replace this with sampling
            # create images from probabilities
            img = torch.exp(data_dist.log_prob(valb))
            # images = data_dist.sample()

            first = img.sum(axis=0)
            second = img.sum(axis=1)

            return torch.cat([first, second], axis=-1)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    # def get_observation(self, num_observation: int) -> torch.Tensor:
    #     """Get observed data for a given observation number
    #     """
    #     if not self.distractors:
    #         path = (
    #             self.path
    #             / "files"
    #             / f"num_observation_{num_observation}"
    #             / "observation.csv"
    #         )
    #         return get_tensor_from_csv(path)
    #     else:
    #         path = (
    #             self.path
    #             / "files"
    #             / f"num_observation_{num_observation}"
    #             / "observation_distractors.csv"
    #         )
    #         return get_tensor_from_csv(path)

    # def _get_transforms(
    #     self,
    #     automatic_transforms_enabled: bool = True,
    #     num_observation: Optional[int] = 1,
    #     observation: Optional[torch.Tensor] = None,
    #     **kwargs: Any,
    # ) -> Dict[str, Any]:
    #     """Gets transforms

    #     Args:
    #         num_observation: Observation number
    #         observation: Instead of passing an observation number, an observation may be
    #             passed directly
    #         automatic_transforms_enabled: If True, will automatically construct
    #             transforms to unconstrained space

    #     Returns:
    #         Dict containing transforms
    #     """
    #     if not self.distractors:
    #         return super()._get_transforms(
    #             automatic_transforms_enabled=automatic_transforms_enabled,
    #             num_observation=num_observation,
    #             observation=observation,
    #             **kwargs,
    #         )
    #     else:
    #         task = norefposterior(distractors=False)
    #         return task._get_transforms(
    #             automatic_transforms_enabled=automatic_transforms_enabled,
    #             num_observation=num_observation,
    #             observation=observation,
    #             **kwargs,
    #         )

    # def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
    #     """Unflattens data into multiple observations
    #     """
    #     if not self.distractors:
    #         return data.reshape(-1, self.num_data, 2)
    #     else:
    #         raise NotImplementedError

    # def _sample_reference_posterior(
    #     self,
    #     num_samples: int,
    #     num_observation: Optional[int] = None,
    #     observation: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """Sample reference posterior for given observation

    #     Args:
    #         num_observation: Observation number
    #         num_samples: Number of samples to generate
    #         observation: Observed data, if None, will be loaded using `num_observation`
    #         kwargs: Passed to run_mcmc

    #     Returns:
    #         Samples from reference posterior
    #     """
    #     from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
    #     from sbibm.algorithms.pytorch.baseline_sir import run as run_sir
    #     from sbibm.algorithms.pytorch.utils.proposal import get_proposal

    #     proposal_samples = run_sir(
    #         task=self,
    #         num_observation=num_observation,
    #         observation=observation,
    #         num_samples=num_samples,
    #         num_simulations=100_000_000,
    #         batch_size=100_000,
    #     )

    #     proposal_dist = get_proposal(
    #         task=self,
    #         samples=proposal_samples,
    #         prior_weight=0.1,
    #         bounded=False,
    #         density_estimator="flow",
    #         flow_model="nsf",
    #     )

    #     return run_rejection(
    #         task=self,
    #         num_observation=num_observation,
    #         observation=observation,
    #         num_samples=num_samples,
    #         batch_size=10_000,
    #         num_batches_without_new_max=1_000,
    #         multiplier_M=1.2,
    #         proposal_dist=proposal_dist,
    #     )

    # def _generate_noise_dist_parameters(self):
    #     import numpy as np

    #     noise_dim = 92
    #     n_noise_comps = 20

    #     rng = np.random
    #     rng.seed(42)

    #     loc = torch.from_numpy(
    #         np.array([15 * rng.normal(size=noise_dim) for i in range(n_noise_comps)])
    #     )

    #     cholesky_factors = [
    #         np.tril(rng.normal(size=(noise_dim, noise_dim)))
    #         + np.diag(np.exp(rng.normal(size=noise_dim)))
    #         for i in range(n_noise_comps)
    #     ]
    #     scale_tril = torch.from_numpy(3 * np.array(cholesky_factors))

    #     mix = pdist.Categorical(torch.ones(n_noise_comps,))
    #     comp = pdist.Independent(
    #         pdist.MultivariateStudentT(df=2, loc=loc, scale_tril=scale_tril), 0,
    #     )
    #     gmm = pdist.MixtureSameFamily(mix, comp)
    #     torch.save(gmm, "files/gmm.torch")

    #     permutation_idx = torch.from_numpy(rng.permutation(noise_dim + 8))
    #     torch.save(permutation_idx, "files/permutation_idx.torch")

    #     torch.manual_seed(42)

    #     for i in range(self.num_observations):
    #         num_observation = i + 1

    #         observation = self.get_observation(num_observation)
    #         noise = gmm.sample().reshape((1, -1)).type(observation.dtype)

    #         observation_and_noise = torch.cat([observation, noise], dim=1)

    #         path = (
    #             self.path
    #             / "files"
    #             / f"num_observation_{num_observation}"
    #             / "observation_distractors.csv"
    #         )
    #         self.dim_data = noise_dim + 8
    #         self.save_data(path, observation_and_noise[:, permutation_idx])


if __name__ == "__main__":

    task = norefposterior()
    # task._generate_noise_dist_parameters()
    task._setup()
