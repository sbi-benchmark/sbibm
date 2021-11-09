import torch

from sbibm.metrics import posterior_mean_error, posterior_variance_ratio


def test_posterior_moments_metrics():

    num_dim = 3
    num_samples = 10000
    dist = torch.distributions.MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))

    mean_err = posterior_mean_error(dist.sample((num_samples,)), dist.sample((num_samples,)))

    assert torch.isclose(mean_err, torch.tensor(0.0), atol=3e-2)

    variance_ratio = posterior_variance_ratio(dist.sample((num_samples,)), dist.sample((num_samples,)))

    assert torch.isclose(variance_ratio, torch.tensor(1.0), atol=1e-1)