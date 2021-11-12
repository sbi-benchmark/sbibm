from typing import Optional, Tuple

import torch
from sbi.inference import MCABC

import sbibm
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
    batch_size: int = 1000,
    save_distances: bool = False,
    kde_bandwidth: Optional[str] = "cv",
    sass: bool = False,
    sass_fraction: float = 0.5,
    sass_feature_expansion_degree: int = 3,
    lra: bool = False,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs REJ-ABC from `sbi`

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
            Fearnhead & Prangle 2012.
        sass_fraction: Fraction of simulation budget to use for sass.
        sass_feature_expansion_degree: Degree of polynomial expansion of the summary
            statistics.
        lra: If True, posterior samples are adjusted with
            linear regression as in Beaumont et al. 2002.
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    assert not (num_top_samples is None and quantile is None and eps is None)

    log = sbibm.get_logger(__name__)
    log.info(f"Running REJ-ABC")

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
