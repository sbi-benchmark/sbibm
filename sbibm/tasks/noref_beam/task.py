from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pyro
import torch
from pyro import distributions as pdist
from torch.distributions import biject_to

from sbibm import get_logger
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv, save_tensor_to_csv


def torch_average(a, weights=None, axis=0):
    """
    emulates np.average interface minimally for pytorch
    (see
    https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy-average)

    Args:
        a : array/tensor to containing data to average
        weights : An array of weights associated with the values in a. Each value in a contributes to the average according to its associated weight.
        axis : Axis or axes along which to average a. The default, axis=0.
    """

    if isinstance(weights, type(None)):
        return a.mean(axis=axis)
    else:
        assert weights.sum() > 0, f"received all 0 weights tensor"
        value = torch.sum(a * weights, axis=axis) / torch.sum(weights, axis=axis)
        return value


def base_coordinate_field(min_axis=-16, max_axis=16, step_width=1.0):
    """returns a torch tensor that contains the coordinates of a regular
    grid between <min_axis> and <max_axis> (at <step_width> from min to max),
    this tensor is broadcasted/cloned <batchsize> times, i.e.

    >>> arr = quadratic_coordinate_field(-3,3)
    >>> arr.shape
    (6,6,2)
     ^^^---- dimensions of max_axis - min_axis, 3-(-3)
    """
    size_axis = max_axis - min_axis
    nsteps = int(size_axis / step_width)

    x = torch.arange(min_axis, max_axis, step_width).detach().float()
    y = torch.arange(min_axis, max_axis, step_width).detach().float()

    xx, yy = torch.meshgrid(x, y)
    val = torch.swapaxes(torch.stack((xx.flatten(), yy.flatten())), 1, 0).float()

    value = val.reshape(nsteps, nsteps, 2)

    return value


def bcast_coordinate_field(base_field, num_samples, swap_to_front=True):
    """utility function that replicates the torch.Tensor <base_field> by <num_samples>
    and moves the last axis to the front

    example:

    >>> arr = torch.from_numpy([[1,2,3],[4,5,6]])
    >>> arr.shape
    (2,3)
    >>> barr = bcast_coordinate_field(arr, 4)
    >>> barr.shape
    (3,2,4)

    Args:
        base_field: the torch tensor to replicate
        num_samples: number of replicates to produce
        swap_to_front: whether to swap the batch axes from the front to the back
    """
    # boadcast to <num_samples> doublicates
    valr_ = torch.broadcast_to(base_field, (num_samples, *base_field.shape)).detach()

    # move axis from position 2 to front
    if swap_to_front:
        value = torch.swapaxes(valr_, 2, 0)
    else:
        value = valr_

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

    given size_axis=max_axis-min_axis, at every point of the image width=size_axis times height=size_axis
    we store the (x,y) coordinate of a regular grid
    so we get:
    valr[0,0] = (-16,-16),
    valr[0,1] = (-16,-15),
    valr[0,2] = (-16,-14)

    Args:
        min_axis: minimum extent of coordinate field
        max_axis: minimum extent of coordinate field
        batch_size: number of replicas to produce
    """

    valr = base_coordinate_field(min_axis, max_axis)

    # broadcast to <batchsize> doublIcates
    valr_ = torch.broadcast_to(valr, (batch_size, *valr.shape)).detach()

    # move axis from position 2 to front
    value = torch.swapaxes(valr_, 2, 0)

    return value


class NorefBeam(Task):
    def __init__(
        self,
        min_axis: int = 0,
        max_axis: int = 100,
        step_width: float = 0.5,
        flood_samples: int = 1 * 1024,
    ):
        """Forward-only simulator (without a reference posterior)

        Inference the parameters of a 2D multivariate normal
        distribution from it's projections onto x and y only
        (surrogate model for a accelerator physics application)

        Args:
            min_axis: minimum extent of the multivariate normal
            max_axis: minimum extent of the multivariate normal
            step_width: number of steps to take between min_axis and max_axis
            flood_samples: number of draws of the binomial wrapping the
        multivariate normal distribution
        """

        self.min_axis = min_axis
        self.max_axis = max_axis
        self.step_width = step_width
        self.nsteps = int((max_axis - min_axis) * (1 / step_width))
        self.flood_samples = flood_samples
        dim_data = 2 * self.nsteps
        name_display = "noref_beam"

        # Observation seeds to use when generating ground truth
        # used to generate the frozen observations (only done once)
        # in case we were to regenerate them
        observation_seeds = list(range(100000, 100000 + 20))

        super().__init__(
            dim_parameters=4,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display=name_display,
            num_observations=20,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        self.prior_params = {
            "low": torch.tensor([20, 20, 1, 1]).float(),
            "high": torch.tensor([80, 80, 15, 15]).float(),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

        self.base_coordinate_field = base_coordinate_field(
            self.min_axis, self.max_axis, self.step_width
        ).detach()

    def get_prior(self) -> Callable:
        def prior(num_samples: int = 1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def _get_transforms(
        self,
        *args,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        This method (as used in the base class) tries to automatically
        construct transformations into unbounded parameter space by inspecting
        the pyro model of this task. Since the output of the task is not equal
        to a sample from a `pyro.sample`-call but rather a reduced version of it
        (due to the `torch.sum` statements in `simulator`) this automatic
        construction cannot work. We therefor override the base class
        behavior by always running the identity_transform.

        The contents of this function were discussed in
        https://github.com/sbi-benchmark/sbibm/pull/34#issuecomment-1006486939
        """

        prior_dist = self.get_prior_dist()
        value = {"parameters": biject_to(prior_dist.support).inv}
        return value

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
            """
            Args:
                parameters: theta parameters coming in (can be batched)

            Details:
                parameters[0]: mean position of "beam" in x
                parameters[1]: mean position of "beam" in y
                parameters[2]: variance of "beam" in x
                parameters[3]: variance of "beam" in y
            """
            batch_size = parameters.shape[0]

            m_ = torch.stack(
                (parameters[:, [0]].squeeze(), parameters[:, [1]].squeeze())
            ).T
            if m_.dim() == 1:
                m_.unsqueeze_(0)

            num_dim = m_.shape[-1]
            # m = torch.broadcast_to(m_, (self.max_axis, self.max_axis, *m_.shape))
            m = m_

            s1 = parameters[:, [2]].squeeze()  # ** 2
            s2 = parameters[:, [3]].squeeze()  # ** 2

            # Note: checking the covariance_matrix for valid inputs
            #       (being positive semidefinite) is expensive, so
            #       `S` needs to be PSD compliant
            #       for the future: consider rotating img for more variability
            S = torch.empty((batch_size, num_dim, num_dim))
            S[..., 0, 0] = s1**2
            S[..., 0, 1] = s1 * s2
            S[..., 1, 0] = 0.0  # s1 * s2
            S[..., 1, 1] = s2**2

            # Add eps to diagonal to ensure PSD
            eps = 0.000001
            S[..., 0, 0] += eps
            S[..., 1, 1] += eps

            # define the probility distribution of our beamspot
            # on a 2D grid (in batches)
            data_dist = pdist.MultivariateNormal(
                m.float(),
                S.float(),
                # `S` is constructed positive semidefinite
                # validation is expensive, so ditch it
                validate_args=False,
            )

            valb = bcast_coordinate_field(
                self.base_coordinate_field, batch_size
            ).detach()

            # create images from log probabilities
            img_ = torch.exp(data_dist.log_prob(valb)).detach()
            img = torch.moveaxis(img_, -1, 0)

            # sample through binomial with fixed prob map
            bdist = pdist.Binomial(total_count=self.flood_samples, probs=img)

            # TODO: should this be a pyro.sample call?
            samples = pyro.sample("data", bdist)

            value = None
            if len(samples.shape) >= 3 and samples.shape == (
                batch_size,
                self.nsteps,
                self.nsteps,
            ):
                # only needed if samples has shape like beam knife-edge scan
                # if pyro model is traced, output of pyro.sample will be different

                # project on the axes
                first = torch.sum(samples, axis=-2)  # along y, onto x
                second = torch.sum(samples, axis=-1)  # along x, onto y

                # concatenate and return
                value = torch.cat([first, second], axis=-1)
            else:
                # if pyro model is traced, samples has shape [batch_size, 2*self.nsteps]
                value = samples
            return value

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)


if __name__ == "__main__":

    log = get_logger(__file__)
    log.warning(
        "[noref_beam] producing observations may result in errors/exceptions thrown!"
    )
    ## run this to generate the `files` infrastructure in this folder
    ## repo/sbibm/sbibm/tasks/noref_beam/files
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
    task = NorefBeam()

    task._setup()
    ## note: the folders mentioned above
