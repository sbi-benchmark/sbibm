import logging
import math
from typing import Callable, Optional

import numpy as np
import torch

from sbibm.tasks.task import Task
from sbibm.third_party.kgof.goftest import KernelSteinTest, bootstrapper_rademacher
from sbibm.third_party.kgof.kernel import KGauss
from sbibm.third_party.kgof.util import meddistance
from sbibm.utils.torch import get_default_device

log = logging.getLogger(__name__)


def ksd(
    task: Task,
    num_observation: int,
    samples: torch.Tensor,
    sig2: Optional[float] = None,
    log: bool = True,
) -> torch.Tensor:
    """Gets `log_prob_grad_fn` from task and runs KSD

    Args:
        task: Task
        num_observation: Observation
        samples: Samples
        sig2: Length scale
        log: Whether to log test result

    Returns:
        The test result is returned
    """
    try:
        device = get_default_device()
        site_name = "parameters"

        log_prob_grad_fn = task._get_log_prob_grad_fn(
            num_observation=num_observation,
            implementation="pyro",
            posterior=True,
            jit_compile=False,
            automatic_transform_enabled=True,
        )

        samples_transformed = task._get_transforms(automatic_transform_enabled=True)[
            site_name
        ](samples)

        def log_prob_grad_numpy(parameters: np.ndarray) -> np.ndarray:
            lpg = log_prob_grad_fn(
                torch.from_numpy(parameters.astype(np.float32)).to(device)
            )
            return lpg.detach().cpu().numpy()

        test_statistic = ksd_gaussian_kernel(
            log_prob_grad_numpy, samples_transformed, sig2=sig2
        )
        if log:
            return math.log(test_statistic)
        else:
            return test_statistic

    except:

        return torch.tensor(float("nan"))


def ksd_gaussian_kernel(
    log_prob_grad_fn: Callable,
    samples: torch.Tensor,
    sig2: Optional[float] = None,
    seed: int = 101,
) -> torch.Tensor:
    """KSD test with `kgof` package

    Args:
        log_prob_grad_fn: Function returning the gradient of the log probability.
            It receives torch.Tensors as inputs and should output a torch.Tensor
            as well.
        samples: Samples for the test as a torch.Tensor

    Returns:
        The test result is returned
    """
    density = UnnormalizedDensityWrapped(log_prob_grad=log_prob_grad_fn)
    samples = DataWrapped(samples)

    if sig2 is None:
        sig2 = meddistance(samples.data(), subsample=1000) ** 2
    else:
        sig2 = float(sig2)

    kernel = KGauss(sig2)

    kstein = KernelSteinTest(
        density,
        kernel,
        bootstrapper=bootstrapper_rademacher,
        alpha=0.01,
        n_simulate=500,
        seed=seed + 1,
    )
    test_result = kstein.perform_test(
        samples, return_simulated_stats=False, return_ustat_gram=False
    )
    # log.info(f"H0 rejected: {test_result['h0_rejected']}")

    # TODO: normalize by sample size?
    return torch.tensor(test_result["test_stat"])


class DataWrapped:
    def __init__(self, data):
        """Wraps `data` such that it can be used with `kgof`"""
        self._data = data

    def data(self):
        return self._data.cpu().numpy()


class UnnormalizedDensityWrapped:
    def __init__(self, log_prob_grad: Callable):
        """Wraps `log_prob_grad` function such that it can be used with `kgof`"""
        self.log_prob_grad = log_prob_grad

    def grad_log(self, X: np.ndarray) -> np.ndarray:
        return self.log_prob_grad(X)
