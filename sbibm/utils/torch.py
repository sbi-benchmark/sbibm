import warnings
from typing import Optional

import numpy as np
import torch
from torch.distributions import Multinomial


def get_default_device() -> torch.device:
    device = torch.ones((1,)).device
    return device


def sample_with_weights(
    values: torch.Tensor, weights: torch.Tensor, num_samples: int
) -> torch.Tensor:
    # define multinomial with weights as probs
    multi = Multinomial(probs=weights)
    # sample num samples, with replacement
    samples = multi.sample(sample_shape=(num_samples,))
    # get indices of success trials
    indices = torch.where(samples)[1]
    # return those indices from trace
    return values[indices]


def sample(
    a: torch.Tensor, num_samples: int, replace: bool = True, seed: Optional[int] = None
) -> torch.Tensor:
    """Sample with or without replacement

    NOTE: See also https://github.com/pytorch/pytorch/issues/16897
    TODO: Dispatch to `torch_sample`?
    """
    if seed is not None:
        np.random.seed(seed)
    num_elements = int(a.shape[0])
    idxs = np.random.choice(num_elements, size=(num_samples,), replace=replace)
    return a[idxs, :]


def choice(*args, **kwargs):
    """Tries to use efficient choice function if available

    Installation:
        pip install "git+https://github.com/LeviViana/torch_sampling#egg=torch_sampling"
    """
    try:
        from torch_sampling import choice

        return choice(*args, **kwargs).long()
    except:
        warnings.warn("Using numpy.random.choice.")
        return choice_numpy(*args, **kwargs).long()


def choice_numpy(
    a, num_samples: int, replace: bool = True, p: torch.Tensor = None
) -> torch.Tensor:
    """Wrap np.random.choice."""

    # Numpy choice needs probs normalized.
    if p is not None:
        if isinstance(p, torch.Tensor):
            p = p.numpy()

        p /= p.sum()

    if isinstance(a, torch.Tensor):
        a = a.numpy()

    samples = np.random.choice(a, num_samples, replace, p)

    return torch.as_tensor(samples, dtype=torch.float32)


def get_log_abs_det_jacobian(
    transform,
    parameters_constrained: torch.Tensor,
    parameters_unconstrained: torch.Tensor,
) -> torch.Tensor:
    """Return log_abs_det_jacobian over batch of parameters.

    Take sum over second dimension if needed, e.g., when torch < 1.8 is used.

    Args:
        transform: transform applied to the parameters
        parameters_constrained:
        parameters_unconstrained:

    Returns:
        torch.Tensor: log_abs_det_jacobian value for each batch entry.
    """

    batch_size, dim_parameters = parameters_constrained.shape

    vals = transform.log_abs_det_jacobian(
        parameters_constrained, parameters_unconstrained
    )

    # For torch < 1.8 log_abs_det_jacobian is returned for each dimension.
    if vals.ndim > 1 and vals.shape[1] == dim_parameters:
        vals = vals.sum(-1)

    assert (
        vals.numel == batch_size
    ), "Mismatch in batch size, took sum over whole batch?"

    return vals
