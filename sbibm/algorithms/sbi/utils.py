import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.distributions.transformed_distribution import TransformedDistribution

from sbibm.utils.nflows import FlowWrapper


def wrap_prior_dist(prior_dist, transforms):
    return TransformedDistribution(prior_dist, transforms)


def wrap_simulator_fn(simulator_fn, transforms):
    return SimulatorWrapper(simulator_fn, transforms)


def wrap_posterior(posterior, transforms):
    return FlowWrapper(posterior, transforms)


class SimulatorWrapper:
    def __init__(self, simulator_fn, transforms):
        self.simulator_fn = simulator_fn
        self.transforms = transforms

    def __call__(self, parameters, *args, **kwargs):
        return self.simulator_fn(self.transforms.inv(parameters))

    @property
    def num_simulations(self):
        return self.simulator_fn.num_simulations


def clip_int(value, minimum, maximum):
    value = int(value)
    minimum = int(minimum)
    maximum = int(maximum)
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def get_sass_transform(theta, x, expansion_degree=1, sample_weight=None):
    """Return semi-automatic summary statitics function.

    Running weighted linear regressin as in
    Fearnhead & Prandle 2012: https://arxiv.org/abs/1004.1112

    Following implementation in
    https://abcpy.readthedocs.io/en/latest/_modules/abcpy/statistics.html#Identity
    and
    https://pythonhosted.org/abcpy/_modules/abcpy/summaryselections.html#Semiautomatic
    """
    expansion = PolynomialFeatures(degree=expansion_degree, include_bias=False)
    # Transform x, remove intercept.
    x_expanded = expansion.fit_transform(x)
    sumstats_map = np.zeros((x_expanded.shape[1], theta.shape[1]))

    for parameter_idx in range(theta.shape[1]):
        regression_model = LinearRegression(fit_intercept=True)
        regression_model.fit(
            X=x_expanded, y=theta[:, parameter_idx], sample_weight=sample_weight
        )
        sumstats_map[:, parameter_idx] = regression_model.coef_

    sumstats_map = torch.tensor(sumstats_map, dtype=torch.float32)

    def sumstats_transform(x):
        x_expanded = torch.tensor(expansion.fit_transform(x), dtype=torch.float32)
        return x_expanded.mm(sumstats_map)

    return sumstats_transform


def run_lra(
    theta: torch.Tensor,
    x: torch.Tensor,
    observation: torch.Tensor,
    sample_weight=None,
    transforms=None,
):
    """Return LRA adjusted parameters."""

    theta_adjusted = transforms(theta)
    for parameter_idx in range(theta.shape[1]):
        regression_model = LinearRegression(fit_intercept=True)
        regression_model.fit(
            X=x,
            y=theta[:, parameter_idx],
            sample_weight=sample_weight,
        )
        theta_adjusted[:, parameter_idx] += regression_model.predict(
            observation.reshape(1, -1)
        )
        theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

    return transforms.inv(theta_adjusted)
