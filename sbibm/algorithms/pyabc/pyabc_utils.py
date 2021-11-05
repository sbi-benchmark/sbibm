import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pyabc
import torch
from sbi.inference import MCABC

import sbibm
from sbibm.tasks.task import Task


class PyAbcSimulator:
    """Wrapper from sbibm task to pyABC.

    pyABC defines its own priors and they are sampled without batch dimension. This
    wrapper defines a call method that takes a single parameter set from a pyABC prior
    and uses the sbibm task simulator to generate the corresponding data and to return
    it in pyABC format.
    """

    def __init__(self, task):
        self.simulator = task.get_simulator()
        self.dim_parameters = task.dim_parameters
        self.name = task.name

    def __call__(self, pyabc_parameter) -> Dict:
        parameters = torch.tensor(
            [[pyabc_parameter[f"param{dim+1}"] for dim in range(self.dim_parameters)]],
            dtype=torch.float32,
        )
        data = self.simulator(parameters).numpy().squeeze()
        return dict(data=data)

    @property
    def __name__(self) -> str:
        return self.name


def wrap_prior(task):
    """Returns a pyABC.Distribution prior given a prior defined on a sbibm task.

    Note: works only for a specific set of priors: Uniform, LogNormal, Normal.
    """
    log = logging.getLogger(__name__)
    log.warn("Will discard any correlations in prior")

    bounds = {}

    prior_cls = str(task.prior_dist)
    if prior_cls == "Independent()":
        prior_cls = str(task.prior_dist.base_dist)

    prior_params = {}
    if "MultivariateNormal" in prior_cls:
        prior_params["m"] = task.prior_params["loc"].numpy()
        if "precision_matrix" in prior_cls:
            prior_params["C"] = np.linalg.inv(
                task.prior_params["precision_matrix"].numpy()
            )
        if "covariance_matrix" in prior_cls:
            prior_params["C"] = task.prior_params["covariance_matrix"].numpy()

        prior_dict = {}
        for dim in range(task.dim_parameters):
            loc = prior_params["m"][dim]
            scale = np.sqrt(prior_params["C"][dim, dim])

            prior_dict[f"param{dim+1}"] = pyabc.RV("norm", loc, scale)
        prior = pyabc.Distribution(**prior_dict)

    elif "LogNormal" in prior_cls:
        # Note the difference in parameterisation between pytorch LogNormal and scipy
        # lognorm:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
        prior_params["s"] = task.prior_params["scale"].numpy()
        prior_params["scale"] = np.exp(task.prior_params["loc"].numpy())

        prior_dict = {}
        for dim in range(task.dim_parameters):
            prior_dict[f"param{dim+1}"] = pyabc.RV(
                "lognorm", s=prior_params["s"][dim], scale=prior_params["scale"][dim]
            )

        prior = pyabc.Distribution(**prior_dict)

    elif "Uniform" in prior_cls:
        prior_params["low"] = task.prior_params["low"].numpy()
        prior_params["high"] = task.prior_params["high"].numpy()

        prior_dict = {}
        for dim in range(task.dim_parameters):
            loc = prior_params["low"][dim]
            scale = prior_params["high"][dim] - loc

            prior_dict[f"param{dim+1}"] = pyabc.RV("uniform", loc, scale)

        prior = pyabc.Distribution(**prior_dict)

    else:
        log.info("No support for prior yet")
        raise NotImplementedError

    return prior


def get_distance(distance: str) -> Callable:
    """Return distance function for pyabc."""

    if distance == "l1":

        def distance_fun(x, y):
            abs_diff = abs(x["data"] - y["data"])
            return np.atleast_1d(abs_diff).mean(axis=-1)

    elif distance == "mse":

        def distance_fun(x, y):
            return np.mean((x["data"] - y["data"]) ** 2, axis=-1)

    elif distance == "l2":

        def distance_fun(x, y):
            return np.linalg.norm(x["data"] - y["data"], axis=-1)

    else:
        raise NotImplementedError(f"Distance '{distance}' not implemented.")

    return distance_fun


def clip_int(value, minimum, maximum):
    value = int(value)
    minimum = int(minimum)
    maximum = int(maximum)
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def run_pyabc(
    task: Task,
    db,
    num_simulations: int,
    observation: np.ndarray,
    pyabc_kwargs: dict,
    distance_str: str = "l2",
    batch_size: int = 1000,
    use_last_pop_samples: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run pyabc SMC with fixed budget and return particles and weights.

    Return previous population or prior samples if budget is exceeded.
    """
    log = sbibm.get_logger(__name__)

    abc = pyabc.ABCSMC(**pyabc_kwargs)
    abc.new(db, {"data": observation})
    history = abc.run(max_total_nr_simulations=num_simulations)
    num_calls = history.total_nr_simulations

    if num_calls < 1.0 * num_simulations:
        (particles_df, weights) = history.get_distribution(t=history.max_t)
        particles = torch.as_tensor(particles_df.values, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)
    else:
        if history.max_t > 0:
            log.info(
                f"Last population exceeded budget by {num_calls - num_simulations}."
            )
            (particles_df, weights) = history.get_distribution(t=history.max_t - 1)
            old_particles = torch.as_tensor(particles_df.values, dtype=torch.float32)
            old_weights = torch.as_tensor(weights, dtype=torch.float32)
            if use_last_pop_samples:
                df = history.get_all_populations()
                num_calls_last_pop = df.samples.values[-1]
                over_budget = num_calls - num_simulations
                proportion_over_budget = over_budget / num_calls_last_pop
                # The proportion over budget needs to be replaced with old particles.
                num_old_particles = int(
                    np.ceil(proportion_over_budget * pyabc_kwargs["population_size"])
                )
                log.info(
                    f"Filling up with {num_old_particles+1} samples from previous population."
                )
                # Combining populations.
                (particles_df, weights) = history.get_distribution(t=history.max_t)
                new_particles = torch.as_tensor(
                    particles_df.values, dtype=torch.float32
                )
                new_weights = torch.as_tensor(weights, dtype=torch.float32)

                particles = torch.zeros_like(old_particles)
                weights = torch.zeros_like(old_weights)
                particles[:num_old_particles] = old_particles[:num_old_particles]
                particles[num_old_particles:] = new_particles[num_old_particles:]
                weights[:num_old_particles] = old_weights[:num_old_particles]
                weights[num_old_particles:] = new_weights[num_old_particles:]
                # Normalize combined weights.
                weights /= weights.sum()
            else:
                log.info("Returning previous population.")
                particles = old_particles
                weights = old_weights
        else:
            log.info("Running REJABC because first population exceeded budget.")
            posterior, _ = run_rejection_abc(
                task,
                num_simulations,
                pyabc_kwargs["population_size"],
                observation=torch.tensor(observation, dtype=torch.float32),
                distance=distance_str,
                batch_size=batch_size,
            )
            particles = posterior._samples
            weights = posterior._log_weights.exp()

    return particles, weights


def run_rejection_abc(
    task: Task,
    num_simulations: int,
    population_size: int,
    observation: Optional[torch.Tensor] = None,
    distance: str = "l2",
    batch_size: int = 1000,
):
    """Return posterior and distances from a ABC with fixed budget."""

    inferer = MCABC(
        simulator=task.get_simulator(max_calls=num_simulations),
        prior=task.get_prior_dist(),
        simulation_batch_size=batch_size,
        distance=distance,
        show_progress_bars=True,
    )
    posterior, distances = inferer(
        x_o=observation,
        num_simulations=num_simulations,
        eps=None,
        quantile=population_size / num_simulations,
        return_distances=True,
    )
    return posterior, distances
