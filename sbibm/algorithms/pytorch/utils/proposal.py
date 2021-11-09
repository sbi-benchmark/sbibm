import math
import time
from typing import Any

import torch
from sbi.utils import get_kde

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.nflows import get_flow, train_flow
from sbibm.utils.torch import choice


def get_proposal(
    task: Task,
    samples: torch.Tensor,
    prior_weight: float = 0.01,
    bounded: bool = True,
    density_estimator: str = "flow",
    flow_model: str = "nsf",
    **kwargs: Any,
) -> torch.Tensor:
    """Gets proposal distribution by performing density estimation on `samples`

    If `prior_weight` > 0., the proposal is defensive, i.e., the prior is mixed in

    Args:
        task: Task instance
        samples: Samples to fit
        prior_weight: Prior weight
        bounded: If True, will automatically transform proposal density to bounded space
        density_estimator: Density estimator
        flow_model: Flow to use if `density_estimator` is `flow`
        kwargs: Passed on to `get_flow` or `get_kde`

    Returns:
        Proposal distribution
    """
    tic = time.time()
    log = sbibm.get_logger(__name__)
    log.info("Get proposal distribution called")

    prior_dist = task.get_prior_dist()
    transform = task._get_transforms(automatic_transforms_enabled=bounded)["parameters"]

    if density_estimator == "flow":
        density_estimator_ = get_flow(
            model=flow_model, dim_distribution=task.dim_parameters, **kwargs
        )
        density_estimator_ = train_flow(
            density_estimator_, samples, transform=transform
        )

    elif density_estimator == "kde":
        density_estimator_ = get_kde(X=samples, transform=transform, **kwargs)

    else:
        raise NotImplementedError

    proposal_dist = DenfensiveProposal(
        dim=task.dim_parameters,
        proposal=density_estimator_,
        prior=prior_dist,
        prior_weight=prior_weight,
    )

    log.info(f"Proposal distribution is set up, took {time.time()-tic:.3f}sec")

    return proposal_dist


class DenfensiveProposal:
    def __init__(self, dim, proposal, prior, prior_weight):
        self.dim = dim

        self.proposal = proposal
        self.prior = prior

        self.prior_weight = prior_weight

        if self.prior_weight == 0.0:
            self.sample = self.sample_proposal
            self.log_prob = self.log_prob_proposal

    def sample(self, num_samples):
        if type(num_samples) in [list, tuple]:
            if len(num_samples) > 1:
                raise NotImplementedError
            num_samples = num_samples[0]

        samples = torch.empty((num_samples, self.dim))
        idxs = choice(
            torch.arange(0, 2),
            num_samples,
            True,
            torch.tensor([self.prior_weight, 1.0 - self.prior_weight]),
        )

        idxs_prior = torch.where(idxs == 0)[0]
        idxs_proposal = torch.where(idxs == 1)[0]

        samples[idxs_prior, :] = self.sample_prior(len(idxs_prior))
        samples[idxs_proposal, :] = self.sample_proposal(len(idxs_proposal))

        return samples

    def sample_prior(self, num_samples):
        return self.prior.sample((num_samples,)).detach()

    def sample_proposal(self, num_samples):
        return self.proposal.sample(num_samples).detach()

    def log_prob(self, parameters):
        return torch.logsumexp(
            torch.stack(
                [
                    math.log(self.prior_weight) + self.log_prob_prior(parameters),
                    math.log(1.0 - self.prior_weight)
                    + self.log_prob_proposal(parameters),
                ]
            ),
            dim=0,
        )

    def log_prob_prior(self, parameters):
        return self.prior.log_prob(parameters).detach()

    def log_prob_proposal(self, parameters):
        return self.proposal.log_prob(parameters).detach()
