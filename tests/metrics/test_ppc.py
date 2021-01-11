import numpy as np
import torch

from sbibm.metrics.ppc import median_distance


def test_median_distance():
    predictive_samples_np = np.random.randn(100, 2)
    predictive_samples = torch.from_numpy(predictive_samples_np)

    observation_np = np.random.randn(1, 2)
    observation = torch.from_numpy(observation_np)

    l2_distance_np = np.linalg.norm(predictive_samples_np - observation_np, axis=1)

    median_np = np.array([np.median(l2_distance_np)]).astype(np.float32)
    median = median_distance(predictive_samples, observation)

    assert torch.allclose(torch.from_numpy(median_np), median)
