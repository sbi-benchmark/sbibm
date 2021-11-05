import numpy as np
import torch


def median_distance(
    predictive_samples: torch.Tensor,
    observation: torch.Tensor,
) -> torch.Tensor:
    """Compute median distance

    Uses NumPy implementation, see [1] for discussion of differences.

    Args:
        predictive_samples: Predictive samples
        observation: Observation

    Returns:
        Median distance

    [1]: https://github.com/pytorch/pytorch/issues/1837
    """
    assert predictive_samples.ndim == 2
    assert observation.ndim == 2

    l2_distance = torch.norm((observation - predictive_samples), dim=-1)
    return torch.tensor([np.median(l2_distance.numpy()).astype(np.float32)])
