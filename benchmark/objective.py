from benchopt import BaseObjective, safe_import_context

import time
from pathlib import Path

with safe_import_context() as import_ctx:
    import torch
    from sbibm.metrics import c2st

# Used to install the sbibm package directly from sources
SBIBM_REPO = Path(__file__).parent.parent


class Objective(BaseObjective):
    r"""Benchmark amortized simulation-based inference (SBI) algorithms.
    """

    name = "SBIBM"
    parameters = {
        'num_posterior_samples': [10_000],
    }
    min_benchopt_version = "1.8"
    sampling_strategy = "run_once"

    # Test configuration: use gaussian_linear task for test and limit the
    # number of samples to fasten C2ST
    test_dataset_name = "gaussian_linear"
    test_config = {
        'num_posterior_samples': 100,
    }

    requirements = [
        f"pip::-e {SBIBM_REPO}",
    ]

    def set_data(self, task, thetas, xs, theta_ref, obs_ref):
        r"""Set the data.

        Input parameters are the output of `Dataset.get_data`.

        Parameters
        ----------
        task: Task
            a task from the SBIBM package
        thetas: torch.Tensor[num_samples, d_params]
            The parameters sampled to train the posterior sampler
        xs: torch.Tensor[num_samples, d_obs]
            The observations sampled to train the posterior sampler
        theta_ref: torch.Tensor[num_reference_samples, d_params]
            The reference posterior samples for the observation `obs_ref`,
            used to evaluate the posterior
        # We can adapt this to have multiple observations.
        obs_ref: torch.Tensor[d_obs]
            The observation for which we want to evaluate the posterior.
        """
        self.task = task
        self.thetas = thetas
        self.xs = xs
        self.theta_ref = theta_ref
        self.obs_ref = obs_ref

    def evaluate_result(self, sample_func):
        """Evaluate a posterior sampler trained by a method

        This function sample `num_posterior_samples` and evaluate them
        with a C2ST test. Also record the sampling time.

        Parameters
        ----------
        sample_func: Callable[[Tuple[int], torch.Tensor], torch.Tensor]
            A callable to sample the posterior associated to one observation.

        Return
        ------
        c2st: float
            The value of the C2ST
        sampling_time: float
            The time taken by the sampler to generate `num_posterior_samples`
            samples from the posterior.
        """

        t_start = time.perf_counter()
        samples = sample_func(
            (self.num_posterior_samples,),
            self.obs_ref
        )
        sampling_time = time.perf_counter() - t_start
        c2st_value = c2st(self.theta_ref, samples).item()

        return dict(
            c2st=c2st_value,
            sampling_time=sampling_time,
        )

    def get_one_result(self):
        r"""Return the same type of output as `Solver.get_result`.

        For testing purposes only.

        Returns
        -------
        sample_func: Callable[[Tuple[int], torch.Tensor], torch.Tensor]
            A callable to sample the posterior associated to one observation.
        """
        return {
            "sample_func": lambda n, x: torch.randn(*n, self.thetas.shape[-1]),
        }

    def get_objective(self):
        """Information to pass to the solvers.

        This information is used by the solvers to compute the result, which
        needs to be compatible with the `evaluate_result` method.
        The API is described in `get_one_result` above.

        Returns
        -------
        Dict
            contains training data and prior required by some solvers (NRE)
            to compute the result: `log_prob` and `sample` functions.
        """
        return dict(thetas=self.thetas, xs=self.xs, task=self.task)
