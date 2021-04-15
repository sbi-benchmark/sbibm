from typing import Optional, Tuple

import numpy as np
import torch
from abcpy.backends import BackendDummy
from abcpy.inferences import RejectionABC
from abcpy.statistics import Identity

import sbibm
from sbibm.algorithms.abcpy.abcpy_utils import (
    ABCpyPrior,
    ABCpySimulator,
    get_distance,
    journal_cleanup_rejABC,
)
from sbibm.tasks.task import Task
from sbibm.utils.io import save_tensor_to_csv
from sbibm.utils.kde import get_kde
from sbibm.utils.torch import sample


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_simulations_per_param: Optional[int] = 1,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_top_samples: Optional[int] = 100,
    quantile: Optional[float] = None,
    eps: Optional[float] = None,
    distance: str = "l2",
    save_distances: bool = False,
    sass: bool = False,
    sass_fraction: float = 0.5,
    sass_feature_expansion_degree: int = 3,
    kde_bandwidth: Optional[str] = None,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs REJ-ABC from `ABCpy`

    ABCpy does not implement LRA post processing. SASS is supported in ABCpy but not yet implemented here.

    Choose one of `num_top_samples`, `quantile`, `eps`.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_simulations_per_param: The number of the total simulations budget to produce for each parameter value;
            used to better estimate the ABC distance from the observation for each parameter value. Useful to
            investigate whether splitting the budget to do multi-simulations per parameter is helpful or not.
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_top_samples: If given, will use `top=True` with num_top_samples
        quantile: Quantile to use
        eps: Epsilon threshold to use
        distance: Distance to use; can be "l2" (Euclidean Distance), "log_reg" (LogReg) or "pen_log_reg" (PenLogReg)
        save_distances: If True, stores distances of samples to disk
        sass: If True, summary statistics are learned as in Fearnhead & Prangle 2012.
            Not yet implemented, left for compatibility with other algorithms.
        sass_fraction: Fraction of simulation budget to use for sass. Unused for now, left for compatibility
        with other algorithms.
        sass_feature_expansion_degree: Degree of polynomial expansion of the summary statistics. Unused for now, only
            left for compatibility with other algorithms.
        kde_bandwidth: If not None, will resample using KDE when necessary, set
            e.g. to "cv" for cross-validated bandwidth selection
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    if sass:
        raise NotImplementedError("SASS not yet implemented")

    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    if (eps is not None) + (num_top_samples is not None) + (quantile is not None) != 1:
        raise RuntimeError(
            "Exactly one of `num_top_samples`, `quantile`, `eps` needs to be specified."
        )

    log = sbibm.get_logger(__name__)
    log.info(f"Running REJ-ABC")

    if observation is None:
        observation = task.get_observation(num_observation)

    if num_top_samples is not None:
        if sass:
            quantile = num_top_samples / (
                num_simulations - int(sass_fraction * num_simulations)
            )
        else:
            quantile = num_top_samples / num_simulations

    # use the dummy backend only (for now)
    backend = BackendDummy()

    # as Rejection ABC works by directly drawing until you get below some threshold -> here we set the threshold to
    # very large value and then we select the best objects later.

    statistics = Identity(
        degree=1, cross=True
    )  # we assume statistics are already computed inside the task
    distance_calc = get_distance(distance, statistics)

    # define prior and model from the task simulator:
    parameters = ABCpyPrior(task)
    model = ABCpySimulator([parameters], task, max_calls=num_simulations)

    # inference
    sampler = RejectionABC(
        root_models=[model], distances=[distance_calc], backend=backend
    )
    journal_standard_ABC = sampler.sample(
        [[np.array(observation)]],
        n_samples=num_simulations // num_simulations_per_param,
        n_samples_per_param=num_simulations_per_param,
        epsilon=10 ** 50,
    )  # set epsilon to very large number
    if (
        eps is None
    ):  # then we use quantile to select the posterior samples from the generated ones
        journal_standard_ABC_reduced = journal_cleanup_rejABC(
            journal_standard_ABC, percentile=quantile * 100
        )
    else:
        journal_standard_ABC_reduced = journal_cleanup_rejABC(
            journal_standard_ABC, threshold=eps
        )

    actual_n_simulations = (
        journal_standard_ABC.number_of_simulations[-1]
        * journal_standard_ABC.configuration["n_samples_per_param"]
    )
    # this takes into account the fact that num_simulations can be not divisible by num_simulations_per_param
    expected_n_simulations = (
        num_simulations // num_simulations_per_param
    ) * num_simulations_per_param
    assert actual_n_simulations == expected_n_simulations

    if save_distances:
        distances = torch.tensor(journal_standard_ABC_reduced.distances[-1])
        save_tensor_to_csv("distances.csv", distances)

    samples = torch.tensor(
        journal_standard_ABC_reduced.get_accepted_parameters()
    ).squeeze()  # that should work

    if kde_bandwidth is not None:
        # fit KDE on data to obtain samples
        log.info(
            f"KDE on {samples.shape[0]} samples with bandwidth option {kde_bandwidth}"
        )
        kde = get_kde(
            samples,
            bandwidth=kde_bandwidth,
        )
        samples = kde.sample(num_samples)
    else:
        # otherwise sample with replacement until you get the num_samples you want
        log.info(f"Sampling {num_samples} samples from trace")
        samples = sample(samples, num_samples=num_samples, replace=True)

    # The latter is None as we cannot compute posterior probabilities with ABCpy (it would be the posterior probability
    # of the true parameter space in case that was available)
    return samples, actual_n_simulations, None
