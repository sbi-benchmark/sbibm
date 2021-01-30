from typing import Optional, Tuple
import numpy as np
import torch
from abcpy.backends import BackendDummy
from abcpy.inferences import RejectionABC
from abcpy.statistics import Identity

import sbibm
from sbibm.algorithms.abcpy.abcpy_utils import ABCpyPrior, get_distance, ABCpySimulator, journal_cleanup_rejABC
from sbibm.tasks.task import Task
from sbibm.utils.io import save_tensor_to_csv


def run(
        task: Task,
        num_samples: int,
        num_simulations: int,
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
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs REJ-ABC from `ABCpy`

    For now, we do not include generating more than one simulation for parameter value; can do that in the future.

    Moreover, we are not using a KDE on the ABC posterior samples in order to obtain a density estimate, as done in the
    pyABC wrap; for this reason we are assuming here that num_samples is the same as num_simulations.

    I believe however we are not returning the expected number of samples, as that would be num_samples, but instead we
    take the given quantile of num_simulations.

    We are also not doing SASS for now, and LRA after sampling.

    Choose one of `num_top_samples`, `quantile`, `eps`. We are actually not using eps for now, as I don't know how to
    combine that with the required simulation budget.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_top_samples: If given, will use `top=True` with num_top_samples
        quantile: Quantile to use
        eps: Epsilon threshold to use
        distance: Distance to use; can be "l2" (Euclidean Distance), "log_reg" (LogReg) or "pen_log_reg" (PenLogReg)
        save_distances: If True, stores distances of samples to disk
        sass: If True, summary statistics are learned as in
            Fearnhead & Prangle 2012.
        sass_fraction: Fraction of simulation budget to use for sass.
        sass_feature_expansion_degree: Degree of polynomial expansion of the summary
            statistics.
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    if sass:
        raise NotImplementedError("SASS not yet implemented")
    if eps is not None:
        raise NotImplementedError

    assert num_simulations == num_samples

    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    assert not (num_top_samples is None and quantile is None and eps is None)

    log = sbibm.get_logger(__name__)
    log.info(f"Running REJ-ABC")

    if observation is None:
        observation = task.get_observation(num_observation)

    if num_top_samples is not None and quantile is None:
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

    statistics = Identity(degree=1, cross=True)  # we assume statistics are already computed inside the task
    distance_calc = get_distance(distance, statistics)

    # define prior and model from the task simulator:
    parameters = ABCpyPrior(task)
    model = ABCpySimulator([parameters], task, num_simulations=num_simulations)

    # inference; not sure how to use random seed
    sampler = RejectionABC(root_models=[model], distances=[distance_calc], backend=backend, seed=None)
    # print("obs", [observation])
    # print("obs", [np.array(observation)])
    journal_standard_ABC = sampler.sample([[np.array(observation)]], n_samples=num_simulations,
                                          n_samples_per_param=1, epsilon=10 ** 50)  # set epsilon to very large number
    journal_standard_ABC_reduced = journal_cleanup_rejABC(journal_standard_ABC, percentile=quantile * 100)

    assert journal_standard_ABC.number_of_simulations[-1] == num_simulations

    if save_distances:
        distances = torch.tensor(journal_standard_ABC_reduced.distances[-1])
        save_tensor_to_csv("/home/lorenzo/Scrivania/OxWaSP/ABC-project/Code/sbibm/distances.csv", distances)

    samples = torch.tensor(journal_standard_ABC_reduced.get_accepted_parameters()).squeeze()  # that should work

    if num_observation is not None:
        # true_parameters = task.get_true_parameters(num_observation=num_observation)
        # log_prob_true_parameters = posterior.log_prob(true_parameters)
        # we can't compute posterior probability with ABCpy
        return samples, journal_standard_ABC.number_of_simulations[-1], None
    else:
        return samples, journal_standard_ABC.number_of_simulations[-1], None
