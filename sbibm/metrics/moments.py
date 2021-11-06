import torch


def mean_error(
    samples: torch.Tensor,
    reference_posterior_samples: torch.Tensor,
    prior_variance: torch.Tensor,
) -> torch.Tensor:
    """Return normalized MSE between posterior means.
    Args:
        samples: Approximate samples
        reference_posterior_samples: Reference posterior samples
        prior_variance: variance of the prior for each dimension
    Returns:
        mean squared error in ratio of the mean difference and prior variance.
    """
    assert samples.ndim == 2
    assert reference_posterior_samples.ndim == 2

    # Normalize with marginal variance of true posterior samples.
    reference_posterior_variance = reference_posterior_samples.var(0)

    squared_error = (
        samples.mean(0) - reference_posterior_samples.mean(0)
    ) ** 2 / reference_posterior_variance

    return torch.mean(squared_error)


def variance_error(
    samples: torch.Tensor,
    reference_posterior_samples: torch.Tensor,
    prior_variance: torch.Tensor,
) -> torch.Tensor:
    """Return normalized MSE between posterior variances.
    Args:
        samples: Approximate samples
        reference_posterior_samples: Reference posterior samples
        prior_variance: variance of the prior for each dimension
    Returns:
        mean squared error in ratio of variance difference and prior variance.
    """
    assert samples.ndim == 2
    assert reference_posterior_samples.ndim == 2

    # Normalize with marginal variance of true posterior samples.
    reference_posterior_variance = reference_posterior_samples.var(0)

    squared_error = (
        samples.var(0) - reference_posterior_samples.var(0)
    ) ** 2 / reference_posterior_variance

    return torch.mean(squared_error)
