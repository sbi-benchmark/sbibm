import logging

import elfi
import numpy as np

from sbibm.tasks.task import Task


def build_prior(task: Task, model: elfi.ElfiModel):
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

        for dim in range(task.dim_parameters):
            loc = prior_params["m"][dim]
            scale = np.sqrt(prior_params["C"][dim, dim])

            elfi.Prior(
                "norm",
                loc,
                scale,
                model=model,
                name=f"parameter_{dim}",
            )

            bounds[f"parameter_{dim}"] = (
                prior_params["m"][dim] - 3.0 * np.sqrt(prior_params["C"][dim, dim]),
                prior_params["m"][dim] + 3.0 * np.sqrt(prior_params["C"][dim, dim]),
            )

    elif "Uniform" in prior_cls:
        prior_params["low"] = task.prior_params["low"].numpy()
        prior_params["high"] = task.prior_params["high"].numpy()

        for dim in range(task.dim_parameters):
            loc = prior_params["low"][dim]
            scale = prior_params["high"][dim] - loc

            elfi.Prior(
                "uniform",
                loc,
                scale,
                model=model,
                name=f"parameter_{dim}",
            )

            bounds[f"parameter_{dim}"] = (
                prior_params["low"][dim],
                prior_params["high"][dim],
            )

    else:
        log.info("No support for prior yet")
        raise NotImplementedError

    return bounds
