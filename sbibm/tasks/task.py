from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyro
import torch

from sbibm.utils.io import get_tensor_from_csv, save_tensor_to_csv
from sbibm.utils.pyro import get_log_prob_fn, get_log_prob_grad_fn


class Task:
    def __init__(
        self,
        dim_data: int,
        dim_parameters: int,
        name: str,
        num_observations: int,
        num_posterior_samples: List[int],
        num_simulations: List[int],
        path: Path,
        name_display: Optional[str] = None,
        num_reference_posterior_samples: int = None,
        observation_seeds: Optional[List[int]] = None,
    ):
        """Base class for tasks.

        Args:
            dim_data: Dimensionality of data.
            dim_parameters: Dimensionality of parameters.
            name: Name of task. Should be the name of the folder in which
                the task is stored. Used with `sbibm.get_task(name)`.
            num_observations: Number of different observations for this task.
            num_posterior_samples: Number of posterior samples to generate.
            num_simulations: List containing number of different simulations to
                run this task for.
            path: Path to folder of task.
            name_display: Display name of task, with correct upper/lower-case
                spelling and spaces. Defaults to `name`.
            num_reference_posterior_samples: Number of reference posterior samples
                to generate for this task. Defaults to `num_posterior_samples`.
            observation_seeds: List of observation seeds to use. Defaults to
                a sequence of length `num_observations`. Override to use specific
                seeds.
        """
        self.dim_data = dim_data
        self.dim_parameters = dim_parameters
        self.name = name
        self.num_observations = num_observations
        self.num_posterior_samples = num_posterior_samples
        self.num_simulations = num_simulations
        self.path = path

        self.name_display = name_display if name_display is not None else name
        self.num_reference_posterior_samples = (
            num_reference_posterior_samples
            if num_reference_posterior_samples is not None
            else num_posterior_samples
        )
        self.observation_seeds = (
            observation_seeds
            if observation_seeds is not None
            else [i + 1000000 for i in range(self.num_observations)]
        )

    @abstractmethod
    def get_prior(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_prior_dist(self) -> torch.distributions.Distribution:
        """Get prior distribution"""
        return self.prior_dist

    def get_prior_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters of prior distribution"""
        return self.prior_params

    def get_labels_data(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"data_{i+1}" for i in range(self.dim_data)]

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"parameter_{i+1}" for i in range(self.dim_parameters)]

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        return get_tensor_from_csv(path)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Get reference posterior samples for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        return get_tensor_from_csv(path)

    @abstractmethod
    def get_simulator(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Get true parameters (parameters that generated the data) for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        return get_tensor_from_csv(path)

    def save_data(self, path: Union[str, Path], data: torch.Tensor):
        """Save data to a given path"""
        save_tensor_to_csv(path, data, self.get_labels_data())

    def save_parameters(self, path: Union[str, Path], parameters: torch.Tensor):
        """Save parameters to a given path"""
        save_tensor_to_csv(path, parameters, self.get_labels_parameters())

    def flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Flattens data

        Data returned by the simulator is always flattened into 2D Tensors
        """
        return data.reshape(-1, self.dim_data)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data

        Tasks that require more than 2 dimensions for output of the simulator (e.g.
        returning images) may override this method.
        """
        return data.reshape(-1, self.dim_data)

    def _get_log_prob_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        posterior: bool = True,
        implementation: str = "pyro",
        **kwargs: Any,
    ) -> Callable:
        """Gets function returning the unnormalized log probability of the posterior or
        likelihood

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will get likelihood instead of posterior
            implementation: Implementation to use, `pyro` or `experimental`
            kwargs: Additional keywords passed to `sbibm.utils.pyro.get_log_prob_fn`

        Returns:
            `log_prob_fn` that returns log probablities as `batch_size`
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)
        assert type(posterior) is bool

        conditioned_model = self._get_pyro_model(
            num_observation=num_observation,
            observation=observation,
            posterior=posterior,
        )

        log_prob_fn, _ = get_log_prob_fn(
            conditioned_model,
            implementation=implementation,
            **kwargs,
        )

        def log_prob_pyro(parameters):
            assert parameters.ndim == 2

            num_parameters = parameters.shape[0]
            if num_parameters == 1:
                return log_prob_fn({"parameters": parameters})
            else:
                log_probs = []
                for i in range(num_parameters):
                    log_probs.append(
                        log_prob_fn({"parameters": parameters[i, :].reshape(1, -1)})
                    )
                return torch.cat(log_probs)

        def log_prob_experimental(parameters):
            return log_prob_fn({"parameters": parameters})

        if implementation == "pyro":
            return log_prob_pyro
        elif implementation == "experimental":
            return log_prob_experimental
        else:
            raise NotImplementedError

    def _get_log_prob_grad_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        posterior: bool = True,
        implementation: str = "pyro",
        **kwargs: Any,
    ) -> Callable:
        """Gets function returning the unnormalized log probability of the posterior

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will get likelihood instead of posterior
            implementation: Implementation to use, `pyro` or `experimental`
            kwargs: Passed to `sbibm.utils.pyro.get_log_prob_grad_fn`

        Returns:
            `log_prob_grad_fn` that returns gradients as `batch_size` x
            `dim_parameter`
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)
        assert type(posterior) is bool
        assert implementation == "pyro"

        conditioned_model = self._get_pyro_model(
            num_observation=num_observation,
            observation=observation,
            posterior=posterior,
        )
        log_prob_grad_fn, _ = get_log_prob_grad_fn(
            conditioned_model,
            implementation=implementation,
            **kwargs,
        )

        def log_prob_grad_pyro(parameters):
            assert parameters.ndim == 2

            num_parameters = parameters.shape[0]
            if num_parameters == 1:
                grads, _ = log_prob_grad_fn({"parameters": parameters})
                return grads["parameters"].reshape(
                    parameters.shape[0], parameters.shape[1]
                )
            else:
                grads = []
                for i in range(num_parameters):
                    grad, _ = log_prob_grad_fn(
                        {"parameters": parameters[i, :].reshape(1, -1)}
                    )
                    grads.append(grad["parameters"].squeeze())
                return torch.stack(grads).reshape(
                    parameters.shape[0], parameters.shape[1]
                )

        if implementation == "pyro":
            return log_prob_grad_pyro
        else:
            raise NotImplementedError

    def _get_transforms(
        self,
        automatic_transforms_enabled: bool = True,
        num_observation: Optional[int] = 1,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Gets transforms

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            automatic_transforms_enabled: If True, will automatically construct
                transforms to unconstrained space

        Returns:
            Dict containing transforms
        """
        conditioned_model = self._get_pyro_model(
            num_observation=num_observation, observation=observation
        )

        _, transforms = get_log_prob_fn(
            conditioned_model,
            automatic_transform_enabled=automatic_transforms_enabled,
        )

        return transforms

    def _get_observation_seed(self, num_observation: int) -> int:
        """Get observation seed for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "observation_seed.csv"
        )
        return int(pd.read_csv(path)["observation_seed"][0])

    def _get_pyro_model(
        self,
        posterior: bool = True,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> Callable:
        """Get model function for use with Pyro

        If `num_observation` or `observation` is passed, the model is conditioned.

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will mask prior which will result in model useful
                for calculating log likelihoods instead of log posterior probabilities
        """
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        prior = self.get_prior()
        simulator = self.get_simulator()

        def model_fn():
            prior_ = pyro.poutine.mask(prior, torch.tensor(posterior))
            return simulator(prior_())

        if observation is not None:
            observation = self.unflatten_data(observation)
            return pyro.condition(model_fn, {"data": observation})
        else:
            return model_fn

    @abstractmethod
    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_samples: Number of samples
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Samples from reference posterior
        """
        raise NotImplementedError

    def _save_observation_seed(self, num_observation: int, observation_seed: int):
        """Save observation seed for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "observation_seed.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[int(observation_seed), int(num_observation)]],
            columns=["observation_seed", "num_observation"],
        ).to_csv(path, index=False)

    def _save_observation(self, num_observation: int, observation: torch.Tensor):
        """Save observed data for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_data(path, observation)

    def _save_reference_posterior_samples(
        self, num_observation: int, reference_posterior_samples: torch.Tensor
    ):
        """Save reference posterior samples for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, reference_posterior_samples)

    def _save_true_parameters(
        self, num_observation: int, true_parameters: torch.Tensor
    ):
        """Save true parameters (parameters that generated the data) for a given observation number"""
        path = (
            self.path
            / "files"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, true_parameters)

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed

        def run(num_observation, observation_seed, **kwargs):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            self._save_true_parameters(num_observation, true_parameters)

            simulator = self.get_simulator()
            observation = simulator(true_parameters)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,
                    **kwargs,
                )
                num_unique = torch.unique(reference_posterior_samples, dim=0).shape[0]
                assert num_unique == self.num_reference_posterior_samples
                self._save_reference_posterior_samples(
                    num_observation,
                    reference_posterior_samples,
                )

        Parallel(n_jobs=n_jobs, verbose=50, backend="loky")(
            delayed(run)(num_observation, observation_seed, **kwargs)
            for num_observation, observation_seed in enumerate(
                self.observation_seeds, start=1
            )
        )
