import pickle
from typing import Optional, Tuple, Union

import torch
from sbi.inference import MCABC
from sbi.utils import KDEWrapper
from torch import Tensor

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.io import save_tensor_to_csv

__DOCSTRING__ = """Runs REJ-ABC from `sbi`

    Choose one of `num_top_samples`, `quantile`, `eps`.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_top_samples: If given, will use `top=True` with num_top_samples
        quantile: Quantile to use
        eps: Epsilon threshold to use
        distance: Distance to use
        batch_size: Batch size for simulator
        save_distances: If True, stores distances of samples to disk
        kde_bandwidth: If not None, will resample using KDE when necessary, set
            e.g. to "cv" for cross-validated bandwidth selection
        sass: If True, summary statistics are learned as in
            Fearnhead & Prangle 2012
            https://doi.org/10.1111/j.1467-9868.2011.01010.x
        sass_fraction: Fraction of simulation budget to use for sass.
        sass_feature_expansion_degree: Degree of polynomial expansion of the summary
            statistics.
        lra: If True, posterior samples are adjusted with
            linear regression as in Beaumont et al. 2002,
            https://doi.org/10.1093/genetics/162.4.2025

    """


def build_posterior(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_top_samples: Optional[int] = 100,
    quantile: Optional[float] = None,
    eps: Optional[float] = None,
    distance: str = "l2",
    batch_size: int = 1000,
    save_distances: bool = False,
    kde_bandwidth: Optional[str] = "cv",
    sass: bool = False,
    sass_fraction: float = 0.5,
    sass_feature_expansion_degree: int = 3,
    lra: bool = False,
) -> Tuple[
    Union[Tuple[Tensor, dict], Tuple[KDEWrapper, dict], Tensor, KDEWrapper], dict
]:
    f"""
    build_posterior method creating the inferred posterior object
    {__DOCSTRING__}

    Returns:
        posterior wrapper, summary dictionary
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    assert not (num_top_samples is None and quantile is None and eps is None)

    log = sbibm.get_logger(__name__)
    log.info(f"Building REJ-ABC posterior")

    prior = task.get_prior_dist()
    simulator = task.get_simulator(max_calls=num_simulations)
    kde = kde_bandwidth is not None
    if observation is None:
        observation = task.get_observation(num_observation)

    if num_top_samples is not None and quantile is None:
        if sass:
            quantile = num_top_samples / (
                num_simulations - int(sass_fraction * num_simulations)
            )
        else:
            quantile = num_top_samples / num_simulations

    inference_method = MCABC(
        simulator=simulator,
        prior=prior,
        simulation_batch_size=batch_size,
        distance=distance,
        show_progress_bars=True,
    )
    # Returns samples or kde posterior in output.
    output, summary = inference_method(
        x_o=observation,
        num_simulations=num_simulations,
        eps=eps,
        quantile=quantile,
        return_summary=True,
        kde=kde,
        kde_kwargs={"bandwidth": kde_bandwidth} if kde else {},
        lra=lra,
        sass=sass,
        sass_expansion_degree=sass_feature_expansion_degree,
        sass_fraction=sass_fraction,
    )

    assert simulator.num_simulations == num_simulations

    if save_distances:
        save_tensor_to_csv("distances.csv", summary["distances"])

    return output, summary


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
    batch_size: int = 1000,
    save_distances: bool = False,
    kde_bandwidth: Optional[str] = "cv",
    sass: bool = False,
    sass_fraction: float = 0.5,
    sass_feature_expansion_degree: int = 3,
    lra: bool = False,
    posterior_path: Optional[str] = "",
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    f"""
    {__DOCSTRING__}
            posterior_path: filesystem location where to store the posterior under
                            (if None, posterior is not saved)

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    assert not (num_top_samples is None and quantile is None and eps is None)

    inkwargs = {k: v for k, v in locals().items() if "posterior_path" not in k}

    log = sbibm.get_logger(__name__)
    log.info(f"Running REJ-ABC")
    simulator = task.get_simulator(max_calls=num_simulations)

    output, summary = build_posterior(**inkwargs)
    kde = kde_bandwidth is not None

    if posterior_path:
        if not kde:
            log.info(
                f"unable to save posterior as non was created, kde = {kde, kde_bandwidth}"
            )
        elif posterior_path is not None:
            log.info(f"storing posterior at {posterior_path}")
            with open(posterior_path, "wb") as ofile:
                pickle.dump(output, ofile)

    if kde:
        kde_posterior = output
        samples = kde_posterior.sample(num_samples)

        # LPTP can only be returned with KDE posterior.
        if num_observation is not None:
            true_parameters = task.get_true_parameters(num_observation=num_observation)
            log_prob_true_parameters = kde_posterior.log_prob(true_parameters)
            return samples, simulator.num_simulations, log_prob_true_parameters
    else:
        samples = output
        return samples, simulator.num_simulations, None
