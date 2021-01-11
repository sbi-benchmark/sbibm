import math
import time
from typing import Any, Optional

import torch
from tqdm.auto import tqdm

import sbibm
from sbibm.algorithms.pytorch.utils.proposal import DenfensiveProposal
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    batch_size: int = 10_000,
    num_batches_without_new_max: int = 1000,
    multiplier_M: float = 1.1,
    proposal_dist: Optional[DenfensiveProposal] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Random samples from rejection sampled posterior as baseline

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        batch_size: Batch size used when finding M
        num_batches_without_new_max: Number of batches that need to be evaluated without
            finding new M before search is stopped
        multiplier_M: Multiplier used when determining M
        proposal_dist: If specified, will be used as a proposal distribution instead
            of prior
        kwargs: Not used

    Returns:
        Random samples from reference posterior
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    tic = time.time()
    log = sbibm.get_logger(__name__)
    log.info("Rejection sampling")

    if "num_simulations" in kwargs:
        log.warn(
            "`num_simulations` was passed as a keyword but will be ignored, since this is a baseline method."
        )

    prior = task.get_prior()
    prior_dist = task.get_prior_dist()

    if proposal_dist is None:
        proposal_dist = prior_dist

    log_prob_fn = task._get_log_prob_fn(
        num_observation=num_observation,
        observation=observation,
        implementation="experimental",
        posterior=True,
    )

    log.info("Finding M")

    log_M = -float("inf")

    parameters = (
        task.get_true_parameters(num_observation=num_observation)
        if num_observation is not None
        else None
    )

    pbar = tqdm(range(num_batches_without_new_max))
    num_batches_cnt = 0
    while num_batches_cnt <= num_batches_without_new_max:
        if parameters is None:
            parameters = prior(num_samples=batch_size)

        log_prob_likelihood_batch = log_prob_fn(parameters)
        log_prob_proposal_batch = proposal_dist.log_prob(parameters)
        log_prob_ratio_max = (log_prob_likelihood_batch - log_prob_proposal_batch).max()

        if log_prob_ratio_max > log_M:
            log_M = log_prob_ratio_max + math.log(multiplier_M)
            num_batches_cnt = 0
            pbar.reset()
            pbar.set_postfix_str(s=f"log(M): {log_M:.3f}", refresh=True)
        else:
            num_batches_cnt += 1
            pbar.update()

        parameters = None
    log.info(f"log(M): {log_M}")

    log.info("Rejection sampling")
    num_sims = 0
    num_accepted = 0
    samples = []
    pbar = tqdm(total=num_samples)
    while num_accepted < num_samples:
        u = torch.rand((batch_size,))
        proposal = proposal_dist.sample((batch_size,))
        probs = log_prob_fn(proposal) - (log_M + proposal_dist.log_prob(proposal))
        num_sims += batch_size

        accept_idxs = torch.where(probs > torch.log(u))[0]
        num_accepted += len(accept_idxs)

        if len(accept_idxs) > 0:
            samples.append(proposal[accept_idxs].detach())
            pbar.update(len(accept_idxs))
            pbar.set_postfix_str(
                s=f"Acceptance rate: {num_accepted/num_sims:.9f}", refresh=True
            )

    pbar.close()

    log.info(f"Acceptance rate: {num_accepted/num_sims:.9f}")
    log.info(f"Finished after {time.time()-tic:.3f} seconds")

    samples = torch.cat(samples)[:num_samples, :]
    assert samples.shape[0] == num_samples

    return samples
