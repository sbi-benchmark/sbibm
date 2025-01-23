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
