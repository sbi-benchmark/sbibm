import time
import warnings
from typing import Any, Dict, Optional

import torch
from pyro.infer.mcmc import HMC, NUTS
from sbi.mcmc.mcmc import MCMC
from sbi.mcmc.slice import Slice

import sbibm
from sbibm.algorithms.pyro.utils.tensorboard import (
    tb_acf,
    tb_ess,
    tb_make_hook_fn,
    tb_marginals,
    tb_posteriors,
    tb_r_hat,
)
from sbibm.tasks.task import Task
from sbibm.utils.tensorboard import tb_make_writer, tb_plot_posterior


def run(
    task: Task,
    num_samples: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_chains: int = 10,
    num_warmup: int = 10000,
    kernel: str = "slice",
    kernel_parameters: Optional[Dict[str, Any]] = None,
    thinning: int = 1,
    diagnostics: bool = True,
    available_cpu: int = 1,
    mp_context: str = "fork",
    jit_compile: bool = False,
    automatic_transforms_enabled: bool = True,
    initial_params: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Runs MCMC using Pyro on potential function

    Produces `num_samples` while accounting for warmup (burn-in) and thinning.

    Note that the actual number of simulations is not controlled for with MCMC since
    algorithms are only used as a reference method in the benchmark.

    MCMC is run on the potential function, which returns the unnormalized
    negative log posterior probability. Note that this requires a tractable likelihood.
    Pyro is used to automatically construct the potential function.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_chains: Number of chains
        num_warmup: Warmup steps, during which parameters of the sampler are adapted.
            Warmup samples are not returned by the algorithm.
        kernel: HMC, NUTS, or Slice
        kernel_parameters: Parameters passed to kernel
        thinning: Amount of thinning to apply, in order to avoid drawing
            correlated samples from the chain
        diagnostics: Flag for diagnostics
        available_cpu: Number of CPUs used to parallelize chains
        mp_context: multiprocessing context, only fork might work
        jit_compile: Just-in-time (JIT) compilation, can yield significant speed ups
        automatic_transforms_enabled: Whether or not to use automatic transforms
        initial_params: Parameters to initialize at

    Returns:
        Samples from posterior
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    tic = time.time()
    log = sbibm.get_logger(__name__)

    hook_fn = None
    if diagnostics:
        log.info(f"MCMC sampling for observation {num_observation}")
        tb_writer, tb_close = tb_make_writer(
            logger=log,
            basepath=f"tensorboard/pyro_{kernel.lower()}/observation_{num_observation}",
        )
        hook_fn = tb_make_hook_fn(tb_writer)

    if "num_simulations" in kwargs:
        warnings.warn(
            "`num_simulations` was passed as a keyword but will be ignored, see docstring for more info."
        )

    # Prepare model and transforms
    conditioned_model = task._get_pyro_model(
        num_observation=num_observation, observation=observation
    )
    transforms = task._get_transforms(
        num_observation=num_observation,
        observation=observation,
        automatic_transforms_enabled=automatic_transforms_enabled,
    )

    kernel_parameters = kernel_parameters if kernel_parameters is not None else {}
    kernel_parameters["jit_compile"] = jit_compile
    kernel_parameters["transforms"] = transforms
    log.info(
        "Using kernel: {name}({parameters})".format(
            name=kernel,
            parameters=",".join([f"{k}={v}" for k, v in kernel_parameters.items()]),
        )
    )
    if kernel.lower() == "nuts":
        mcmc_kernel = NUTS(model=conditioned_model, **kernel_parameters)

    elif kernel.lower() == "hmc":
        mcmc_kernel = HMC(model=conditioned_model, **kernel_parameters)

    elif kernel.lower() == "slice":
        mcmc_kernel = Slice(model=conditioned_model, **kernel_parameters)

    else:
        raise NotImplementedError

    if initial_params is not None:
        site_name = "parameters"
        initial_params = {site_name: transforms[site_name](initial_params)}
    else:
        initial_params = None

    mcmc_parameters = {
        "num_chains": num_chains,
        "num_samples": thinning * num_samples,
        "warmup_steps": num_warmup,
        "available_cpu": available_cpu,
        "initial_params": initial_params,
    }
    log.info(
        "Calling MCMC with: MCMC({name}_kernel, {parameters})".format(
            name=kernel,
            parameters=",".join([f"{k}={v}" for k, v in mcmc_parameters.items()]),
        )
    )

    mcmc = MCMC(mcmc_kernel, hook_fn=hook_fn, **mcmc_parameters)
    mcmc.run()

    toc = time.time()
    log.info(f"Finished MCMC after {toc-tic:.3f} seconds")
    log.info(f"Automatic transforms {mcmc.transforms}")

    log.info(f"Apply thinning of {thinning}")
    mcmc._samples = {"parameters": mcmc._samples["parameters"][:, ::thinning, :]}

    num_samples_available = (
        mcmc._samples["parameters"].shape[0] * mcmc._samples["parameters"].shape[1]
    )
    if num_samples_available < num_samples:
        warnings.warn("Some samples will be included multiple times")
        samples = mcmc.get_samples(num_samples=num_samples, group_by_chain=False)[
            "parameters"
        ].squeeze()
    else:
        samples = mcmc.get_samples(group_by_chain=False)["parameters"].squeeze()
        idx = torch.randperm(samples.shape[0])[:num_samples]
        samples = samples[idx, :]

    assert samples.shape[0] == num_samples

    if diagnostics:
        mcmc.summary()
        tb_ess(tb_writer, mcmc)
        tb_r_hat(tb_writer, mcmc)
        tb_marginals(tb_writer, mcmc)
        tb_acf(tb_writer, mcmc)
        tb_posteriors(tb_writer, mcmc)
        tb_plot_posterior(tb_writer, samples, tag="posterior/final")
        tb_close()

    return samples
