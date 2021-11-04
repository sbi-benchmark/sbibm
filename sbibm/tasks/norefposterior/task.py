from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pyro
import torch
from pyro import distributions as pdist

from sbibm import get_logger
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv, save_tensor_to_csv


def torch_average(a, weights=None, axis=0):

    if isinstance(weights, type(None)):
        return a.mean(axis=axis)
    else:
        assert weights.sum() > 0, f"received all 0 weights tensor"
        value = torch.sum(a * weights, axis=axis) / torch.sum(weights, axis=axis)
        return value


def base_coordinate_field(min_axis=-16, max_axis=16):
    """returns a torch tensor that contains the coordinates of a regular
    grid between <min_axis> and <max_axis> broadcasted/cloned
    <batchsize> times, i.e.
    >>> arr = quadratic_coordinate_field(-3,3)
    >>> arr.shape
    (6,6,2)
     ^^^---- dimensions of max_axis - min_axis, 3-(-3)
    """
    size_axis = max_axis - min_axis

    x = torch.arange(min_axis, max_axis).detach().float()
    y = torch.arange(min_axis, max_axis).detach().float()

    xx, yy = torch.meshgrid(x, y)
    val = torch.swapaxes(torch.stack((xx.flatten(), yy.flatten())), 1, 0).float()

    value = val.reshape(size_axis, size_axis, 2)

    return value


def bcast_coordinate_field(base_field, num_samples):

    # boadcast to <num_samples> doublicates
    valr_ = torch.broadcast_to(base_field, (num_samples, *base_field.shape)).detach()

    # move axis from position 2 to front
    value = torch.swapaxes(valr_, 2, 0)

    return value


def quadratic_coordinate_field(min_axis=-16, max_axis=16, batch_size=32):
    """returns a torch tensor that contains the coordinates of a regular
    grid between <min_axis> and <max_axis> broadcasted/cloned
    <batchsize> times, i.e.
    >>> arr = quadratic_coordinate_field(-3,3,4)
    >>> arr.shape
    #    --- batch_size
    #    v
    (6,6,4,2)
    #^ ^
    #| |
    #------- dimensions of max_axis - min_axis, 3-(-3)
    """

    valr = base_coordinate_field(min_axis, max_axis)

    # at every point of the image w=size_axis x w=size_axis
    # we store the (x,y) coordinate of a regular grid
    # so we get:
    # valr[0,0] = (-16,-16),
    # valr[0,1] = (-16,-15),
    # valr[0,2] = (-16,-14)

    # broadcast to <batchsize> doublicates
    valr_ = torch.broadcast_to(valr, (batch_size, *valr.shape)).detach()

    # move axis from position 2 to front
    value = torch.swapaxes(valr_, 2, 0)

    return value


class norefposterior(Task):
    def __init__(self, min_axis=0, max_axis=200, flood_samples=1 * 1024):
        """Forward-only simulator (without a reference posterior)"""

        self.min_axis = min_axis
        self.max_axis = max_axis
        self.flood_samples = flood_samples
        dim_data = 2 * self.max_axis
        name_display = "norefposterior"

        # TODO: not clear what purpose these serve
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
            "low": torch.tensor([20, 20, 5, 5]).float(),
            "high": torch.tensor([80, 80, 15, 15]).float(),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

        self.base_coordinate_field = base_coordinate_field(
            self.min_axis, self.max_axis
        ).detach()

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

            s1 = parameters[:, [2]].squeeze()  # ** 2
            s2 = parameters[:, [3]].squeeze()  # ** 2

            # Note: checking the covariance_matrix for valid inputs
            #       (being positive semidefinite) is expense, so
            #       `S` needs to be PSD compliant
            #       for the future: consider rotating img for more variability
            S = torch.empty((self.max_axis, self.max_axis, num_samples, 2, 2))
            S[..., 0, 0] = s1 ** 2
            S[..., 0, 1] = s1 * s2
            S[..., 1, 0] = 0.0  # s1 * s2
            S[..., 1, 1] = s2 ** 2

            # Add eps to diagonal to ensure PSD
            eps = 0.000001
            S[..., 0, 0] += eps
            S[..., 1, 1] += eps

            assert S.shape == (
                self.max_axis,
                self.max_axis,
                num_samples,
                2,
                2,
            ), f"{name_display} :: cov matrix {S.shape} != expectation"
            assert m.shape == (
                self.max_axis,
                self.max_axis,
                num_samples,
                2,
            ), f"{name_display} :: mean vector {m.shape} != expectation"

            # define the probility distribution of our beamspot
            # on a 2D grid (in batches)
            data_dist = pdist.MultivariateNormal(m.float(), S.float(),
                                                 # `S` is constructed positive semidefinite
                                                 # validation is expensive
                                                 validate_args = False
                                                 )

            valb = bcast_coordinate_field(
                self.base_coordinate_field, num_samples
            ).detach()
            # valb = quadratic_coordinate_field(self.min_axis, self.max_axis, num_samples).detach()

            # create images from log probabilities
            img = torch.exp(data_dist.log_prob(valb)).detach()

            # sample through binomial with fixed prob map
            bdist = pdist.Binomial(total_count=self.flood_samples, probs=img)

            # TODO: should this be a pyro.sample call?
            samples = bdist.sample()

            # project on the axes
            first = torch.sum(samples, axis=0)
            second = torch.sum(samples, axis=1)

            # concatenate and return
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


if __name__ == "__main__":

    log = get_logger(__file__)
    log.warning(
        "[norefposterior] producing observations may result in errors/exceptions thrown!"
    )
    ## run this to generate the `files` infrastructure in this folder
    ## repo/sbibm/sbibm/tasks/norefposterior/files
    ## ├── num_observation_1
    ## ├── num_observation_10
    ## ├── num_observation_2
    ## ├── num_observation_3
    ## ├── num_observation_4
    ## ├── num_observation_5
    ## ├── num_observation_6
    ## ├── num_observation_7
    ## ├── num_observation_8
    ## └── num_observation_9
    task = norefposterior()

    task._setup()
    ## note: the folders mentioned above
