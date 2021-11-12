from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pyro
import pyro.distributions as pdist
import torch

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv
from sbibm.utils.torch import get_default_device


class BernoulliGLM(Task):
    def __init__(self, summary="sufficient"):
        """Bernoulli GLM"""
        self.summary = summary
        if self.summary == "sufficient":
            dim_data = 10
            name = "bernoulli_glm"
            name_display = "Bernoulli GLM"
            self.raw = False
        elif self.summary == "raw":
            dim_data = 100
            self.raw = True
            name = "bernoulli_glm_raw"
            name_display = "Bernoulli GLM Raw"
        else:
            raise NotImplementedError

        super().__init__(
            dim_parameters=10,
            dim_data=dim_data,
            name=name,
            name_display=name_display,
            num_simulations=[1000, 10000, 100000, 1000000],
            num_posterior_samples=10000,
            num_observations=10,
            path=Path(__file__).parent.absolute(),
        )

        self.stimulus = {
            "dt": 1,  # timestep
            "duration": 100,  # duration of input stimulus
            "seed": 42,  # seperate seed to freeze noise on input current
        }

        # Prior on offset and filter
        # Smoothness in filter encouraged by penalyzing 2nd order differences
        M = self.dim_parameters - 1
        D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
        F = torch.matmul(D, D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
        Binv = torch.zeros(size=(M + 1, M + 1))
        Binv[0, 0] = 0.5  # offset
        Binv[1:, 1:] = torch.matmul(F.T, F)  # filter

        self.prior_params = {"loc": torch.zeros((M + 1,)), "precision_matrix": Binv}
        self.prior_dist = pdist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """
        device = get_default_device()

        stimulus_I = torch.load(self.path / "files" / "stimulus_I.pt").to(device)
        design_matrix = torch.load(self.path / "files" / "design_matrix.pt").to(device)

        def simulator(
            parameters: torch.Tensor, return_both: bool = False
        ) -> torch.Tensor:
            """Simulates model for given parameters

            If `return_both` is True, will additionally return spike train not reduced to summary features
            """

            data = []
            data_raw = []
            for b in range(parameters.shape[0]):
                # Simulate GLM
                psi = torch.matmul(design_matrix, parameters[b, :])
                z = 1 / (1 + torch.exp(-psi))
                y = (torch.rand(design_matrix.shape[0]) < z).float()

                # Calculate summary statistics
                num_spikes = torch.sum(y).unsqueeze(0)
                sta = torch.nn.functional.conv1d(
                    y.reshape(1, 1, -1), stimulus_I.reshape(1, 1, -1), padding=8
                ).squeeze()[-9:]
                data.append(torch.cat((num_spikes, sta)))

                if self.raw or return_both:
                    data_raw.append(y)

            if not return_both:
                if not self.raw:
                    return torch.stack(data)
                else:
                    return torch.stack(data_raw)
            else:
                return torch.stack(data), torch.stack(data_raw)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        if not self.raw:
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation.csv"
            )
            return get_tensor_from_csv(path)
        else:
            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation_raw.csv"
            )
            return get_tensor_from_csv(path)

    def flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Flattens data

        Data returned by the simulator is always flattened into 2D Tensors
        """
        if type(data) == tuple:
            return data
        else:
            return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
    ) -> torch.Tensor:
        from pypolyagamma import PyPolyaGamma
        from tqdm import tqdm

        self.dim_data = 10
        # stimulus_I = torch.load(self.path / "files" / "stimulus_I.pt")
        design_matrix = torch.load(self.path / "files" / "design_matrix.pt")
        true_parameters = self.get_true_parameters(num_observation)
        self.raw = True
        observation_raw = self.get_observation(num_observation)
        self.raw = False

        mcmc_num_samples_warmup = 25000
        mcmc_thinning = 25
        mcmc_num_samples = mcmc_num_samples_warmup + mcmc_thinning * num_samples

        pg = PyPolyaGamma()
        X = design_matrix.numpy()
        obs = observation_raw.numpy()
        Binv = self.prior_params["precision_matrix"].numpy()

        sample = true_parameters.numpy().reshape(-1)  # Init at true parameters
        samples = []
        for j in tqdm(range(mcmc_num_samples)):
            psi = np.dot(X, sample)
            w = np.array([pg.pgdraw(1, b) for b in psi])
            O = np.diag(w)  # noqa: E741
            V = np.linalg.inv(np.dot(np.dot(X.T, O), X) + Binv)
            m = np.dot(V, np.dot(X.T, obs.reshape(-1) - 1 * 0.5))
            sample = np.random.multivariate_normal(np.ravel(m), V)
            samples.append(sample)
        samples = np.asarray(samples).astype(np.float32)
        samples_subset = samples[mcmc_num_samples_warmup::mcmc_thinning, :]

        reference_posterior_samples = torch.from_numpy(samples_subset)

        return reference_posterior_samples

    def _setup(self, regenerate_stimulus=False):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.
        Re-executing will overwrite existing files.

        Reference samples are constructed using Polya-Gamma MCMC. The sampler consists of two iterative Gibbs updates:
        1. sample auxiliary variables: w ~ PG(N, psi)
        2. sample parameters: beta ~ N(m, V); V = inv(X'O X + Binv), m = V*(X'k), k = y - N/2

        Note that running this method requires pypolyagamma, see https://github.com/slinderman/pypolyagamma
        for installation instructions.

        There is an open issue leading to errors on pip install, see:
        https://github.com/slinderman/pypolyagamma/issues/36

        Manual installation with the following steps succeeded:
        pip install cython==0.28
        git clone git@github.com:slinderman/pypolyagamma.git
        cd pypolyagamma
        pip install -e .

        Takes about 1-2 minutes on a laptop per observation
        """
        # Generate input stimulus (same across all observations)
        # Stimulus is Gaussian white noise ~N(0, 1)
        if regenerate_stimulus:
            stimulus_t = torch.arange(
                0, self.stimulus["duration"], self.stimulus["dt"], dtype=torch.float32
            )
            path = self.path / "files" / "stimulus_t.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stimulus_t, path)
            stimulus_I = torch.from_numpy(
                np.random.RandomState(self.stimulus["seed"])
                .randn(len(stimulus_t))
                .reshape(-1)
                .astype(np.float32)
            )
            path = self.path / "files" / "stimulus_I.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stimulus_I, path)

            # Build design matrix X, such that X * h returns convolution of x with filter h
            # Including linear offset by first element
            design_matrix = torch.zeros(size=(len(stimulus_t), self.dim_parameters - 1))
            for j in range(self.dim_parameters - 1):
                design_matrix[j:, j] = stimulus_I[0 : len(stimulus_t) - j]
            design_matrix = torch.cat(
                (torch.ones(size=(len(stimulus_t), 1)), design_matrix), axis=1
            )
            path = self.path / "files" / "design_matrix.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(design_matrix, path)

        observation_seeds = np.arange(self.num_observations, dtype=np.int) + int(1e6)
        for num_observation, observation_seed in enumerate(observation_seeds, start=1):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            self._save_true_parameters(num_observation, true_parameters)

            simulator = self.get_simulator()
            observation, observation_raw = simulator(true_parameters, return_both=True)
            self._save_observation(num_observation, observation)

            path = (
                self.path
                / "files"
                / f"num_observation_{num_observation}"
                / "observation_raw.csv"
            )
            self.dim_data = 100
            self.save_data(path, observation_raw)
            self.dim_data = 10

            reference_posterior_samples = self._sample_reference_posterior(
                num_samples=10_000, num_observation=num_observation
            )

            self._save_reference_posterior_samples(
                num_observation, reference_posterior_samples
            )


if __name__ == "__main__":
    task = BernoulliGLM()
    task._setup()
