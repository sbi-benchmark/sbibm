import torch


def posterior_mean_error(
    samples: torch.Tensor,
    reference_posterior_samples: torch.Tensor,
) -> torch.Tensor:
    """Return absolute error between posterior means normalized by true std.
    Args:
        samples: Approximate samples
        reference_posterior_samples: Reference posterior samples
    Returns:
        absolute error in posterior mean, normalized by std, averaged over dimensions.
    """
    assert samples.ndim == 2
    assert reference_posterior_samples.ndim == 2
    abs_error_per_dim = (
        samples.mean(0) - reference_posterior_samples.mean(0)
    ) / reference_posterior_samples.std(0)

    return torch.mean(abs_error_per_dim)


def posterior_variance_ratio(
    samples: torch.Tensor,
    reference_posterior_samples: torch.Tensor,
) -> torch.Tensor:
    """Return ratio of approximate and true variance, averaged over dimensions.
    Args:
        samples: Approximate samples
        reference_posterior_samples: Reference posterior samples
    Returns:
        ratio of approximate and true posterior variance, averaged over dimensions.
    """
    assert samples.ndim == 2
    assert reference_posterior_samples.ndim == 2

    ratio_per_dim = samples.var(0) / reference_posterior_samples.var(0)

    return torch.mean(ratio_per_dim)
