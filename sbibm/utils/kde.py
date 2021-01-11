from typing import Optional, Union

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch import distributions as dist

transform_types = Optional[
    Union[
        torch.distributions.transforms.Transform,
        torch.distributions.transforms.ComposeTransform,
    ]
]


def get_kde(
    X: torch.Tensor,
    bandwidth: str = "cv",
    transform: transform_types = None,
    verbose: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> KernelDensity:
    """Get KDE estimator with selected bandwidth

    Args:
        X: Samples
        bandwidth: Bandwidth method
        transform: Optional transform
        sample_weight: Sample weights attached to the data 
        verbose: Verbosity level

    References:
    [1]: https://github.com/scikit-learn/scikit-learn/blob/0303fca35e32add9d7346dcb2e0e697d4e68706f/sklearn/neighbors/kde.py
    """
    if transform is None or not transform:
        transform = dist.transforms.identity_transform

    X = transform(X).numpy()

    algorithm = "auto"
    kernel = "gaussian"
    metric = "euclidean"
    atol = 0
    rtol = 0
    breadth_first = True
    leaf_size = 40
    metric_params = None

    if bandwidth == "scott":
        bandwidth_selected = X.shape[0] ** (-1.0 / (X.shape[1] + 4))
    elif bandwidth == "silvermann":
        bandwidth_selected = (X.shape[0] * (X.shape[1] + 2) / 4.0) ** (
            -1.0 / (X.shape[1] + 4)
        )
    elif bandwidth == "cv":
        steps = 10
        lower = 0.1 * X.std()
        upper = 0.5 * X.std()
        current_best = -10000000

        for _ in range(5):
            bandwidth_range = np.linspace(lower, upper, steps)
            grid = GridSearchCV(
                KernelDensity(
                    kernel=kernel,
                    algorithm=algorithm,
                    metric=metric,
                    atol=atol,
                    rtol=rtol,
                    breadth_first=breadth_first,
                    leaf_size=leaf_size,
                    metric_params=metric_params,
                ),
                {"bandwidth": bandwidth_range},
                cv=20,
            )
            grid.fit(X)

            if abs(current_best - grid.best_score_) > 0.001:
                current_best = grid.best_score_
            else:
                break

            second_best_index = list(grid.cv_results_["rank_test_score"]).index(2)

            if (grid.best_index_ == 0) or (grid.best_index_ == steps):
                diff = (lower - upper) / steps
                lower = grid.best_index_ - diff
                upper = grid.best_index_ + diff
            else:
                upper = bandwidth_range[second_best_index]
                lower = bandwidth_range[grid.best_index_]

                if upper < lower:
                    upper, lower = lower, upper

        bandwidth_selected = grid.best_params_["bandwidth"]
    elif bandwidth > 0:
        bandwidth_selected = float(bandwidth)
    else:
        raise ValueError("bandwidth must be positive, scott, silvermann or cv")

    kde = KernelDensity(
        kernel=kernel,
        algorithm=algorithm,
        metric=metric,
        atol=atol,
        rtol=rtol,
        breadth_first=breadth_first,
        leaf_size=leaf_size,
        metric_params=metric_params,
        bandwidth=bandwidth_selected,
    )
    kde.fit(X, sample_weight=sample_weight)

    return KDEWrapper(kde, transform)


class KDEWrapper:
    def __init__(self, kde, transform):
        self.kde = kde
        self.transform = transform

    def sample(self, *args, **kwargs):
        Y = torch.from_numpy(self.kde.sample(*args, **kwargs).astype(np.float32))
        return self.transform.inv(Y)

    def log_prob(self, parameters_constrained):
        parameters_unconstrained = self.transform(parameters_constrained)
        log_probs = torch.from_numpy(
            self.kde.score_samples(parameters_unconstrained.numpy()).astype(np.float32)
        )
        log_probs += torch.sum(
            self.transform.log_abs_det_jacobian(
                parameters_constrained, parameters_unconstrained
            ),
            axis=1,
        )
        return log_probs
