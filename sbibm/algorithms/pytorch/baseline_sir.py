import time
from typing import Any, Optional

import torch
from tqdm.auto import tqdm

import sbibm
from sbibm.algorithms.pytorch.utils.proposal import DenfensiveProposal
from sbibm.tasks.task import Task
from sbibm.utils.torch import choice


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    batch_size: int = 100000,
    proposal_dist: Optional[DenfensiveProposal] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Random samples from Sequential Importance Resampling (SIR) as a baseline

    SIR is also referred to as weighted bootstrap [1]. The prior is used as a proposal,
    so that the weights become the likelihood, this has also been referred to as
    likelihood weighting in the literature.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        batch_size: Batch size for simulations
        proposal_dist: If specified, will be used as a proposal distribution instead
            of prior
        kwargs: Not used

    Returns:
        Random samples from reference posterior

    [1] A. F. M. Smith and A. E. Gelfand. Bayesian statistics without tears: a
    sampling-resampling perspective. The American Statistician, 46(2):84-88, 1992.
    doi:10.1080/00031305.1992.10475856.
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    tic = time.time()
    log = sbibm.get_logger(__name__)
    log.info("Sequential Importance Resampling (SIR)")

    prior_dist = task.get_prior_dist()

    if proposal_dist is None:
        proposal_dist = prior_dist

    log_prob_fn = task._get_log_prob_fn(
        num_observation=num_observation,
        observation=observation,
        implementation="experimental",
        posterior=True,
    )

    batch_size = min(batch_size, num_simulations)
    num_batches = int(num_simulations / batch_size)

    particles = []
    log_weights = []
    for i in tqdm(range(num_batches)):
        batch_draws = proposal_dist.sample((batch_size,))
        log_weights.append(
            log_prob_fn(batch_draws) - proposal_dist.log_prob(batch_draws)
        )
        particles.append(batch_draws)
    log.info("Finished sampling")

    particles = torch.cat(particles)
    log_weights = torch.cat(log_weights)
    probs = torch.exp(log_weights.view(-1))
    probs /= probs.sum()

    indices = torch.arange(0, len(probs))
    idxs = choice(indices, num_samples, True, probs)
    samples = particles[idxs, :]
    log.info("Finished resampling")

    num_unique = torch.unique(samples, dim=0).shape[0]
    log.info(f"Unique particles: {num_unique} out of {len(samples)}")

    toc = time.time()
    log.info(f"Finished after {toc-tic:.3f} seconds")

    return samples
