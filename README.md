[![PyPI
version](https://badge.fury.io/py/sbibm.svg)](https://badge.fury.io/py/sbibm) [![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sbi-benchmark/sbibm/blob/master/CONTRIBUTING.md) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)

# Simulation-Based Inference Benchmark

This repository contains a simulation-based inference benchmark framework, `sbibm`, which we describe in the [associated manuscript "Benchmarking Simulation-based Inference"](https://github.com/mackelab/sbibm#Citation). The benchmark framework includes tasks, reference posteriors, metrics, plotting, and integrations with SBI toolboxes. The framework is designed to be highly extensible and easily used in new research projects: For each benchmark task, prior, simulator, and reference posteriors are exposed, so that `sbibm` can be used easily in research code, as we demonstrate below. 

In order to emphasize that `sbibm` can be used independently of any particular analysis pipeline, we split the code for reproducing the experiments of the manuscript into a seperate repository hosted at [github.com/sbi-benchmark/benchmarking_sbi](https://github.com/sbi-benchmark/benchmarking_sbi). Besides the pipeline to reproduce the manuscripts' experiments, full results including dataframes for quick comparisons are hosted in that repository.

If you have questions or comments, please do not hesitate [to contact us](mailto:mail@jan-matthis.de) or [open an issue](https://github.com/sbi-benchmark/sbibm/issues). We would be very glad [about contributions](CONTRIBUTE.md), e.g., new tasks, novel metrics, or wrappers for other SBI toolboxes.


## Installation

Assuming you have a working Python environment, simply install `sbibm` via `pip`:
```commandline
$ pip install sbibm
```

ODE based models (currently SIR and Lotka-Volterra models) use [Julia](https://julialang.org) via [`diffeqtorch`](https://github.com/sbi-benchmark/diffeqtorch). If you are planning to use these tasks, please additionally follow the [installation instructions of `diffeqtorch`](https://github.com/sbi-benchmark/diffeqtorch#installation). If you are not planning to simulate these tasks for now, you can skip this step.


## Tasks

You can then see the list of available tasks by calling `sbibm.get_available_tasks()`. If we wanted to use, say, the `slcp` task, we can load it using `sbibm.get_task`, as in:

```python
import sbibm
task = sbibm.get_task("slcp")
```

Next, we might want to get `prior` and `simulator`:

```python
prior = task.get_prior()
simulator = task.get_simulator()
```

If we call `prior()` we get a single draw from the prior distribution. `num_samples` can be provided as an optional argument. The following would generate 100 samples from the simulator: 
```python
thetas = prior(num_samples=100)
xs = simulator(thetas)
```

`xs` is a `torch.Tensor` with shape `(100, 8)`, since for SLCP the data is eight-dimensional. Note that if required, conversion to and from `torch.Tensor` is very easy: Convert to a numpy array using `.numpy()`, e.g., `xs.numpy()`. For the reverse, use `torch.from_numpy()` on a numpy array.

Some algorithms might require evaluating the pdf of the prior distribution, which can be obtained as a [`torch.Distribution` instance](https://pytorch.org/docs/stable/distributions.html) using `task.get_prior_dist()`, which exposes `log_prob` and `sample` methods. The parameters of the prior can be picked up as a dictionary as parameters using `task.get_prior_params()`.

For each task, the benchmark contains 10 observations and respective reference posteriors samples. To fetch the first observation and respective reference posterior samples:
```python
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)
```

Every tasks has a couple of informative attributes, including:

```python
task.dim_data               # dimensionality data, here: 8
task.dim_parameters         # dimensionality parameters, here: 5
task.num_observations       # number of different observations x_o available, here: 10
task.name                   # name: slcp
task.name_display           # name_display: SLCP
```

Finally, if you want to have a look at the source code of the task, take a look in `sbibm/tasks/slcp/task.py`. If you wanted to implement a new task, we would recommend modelling them after the existing ones. You will see that each task has a private `_setup` method that was used to generate the reference posterior samples. 


## Algorithms

As mentioned in the intro, `sbibm` wraps a number of third-party packages to run various algorithms. We found it easiest to give each algorithm the same interface: In general, each algorithm specifies a `run` function that gets `task` and hyperparameters as arguments, and eventually returns the required `num_posterior_samples`. That way, one can simply import the run function of an algorithm, tune it on any given task, and return metrics on the returned samples. Wrappers for external toolboxes implementing algorithms are in the subfolder `sbibm/algorithms`. Currently, integrations with [`sbi`](https://www.mackelab.org/sbi/), [`pyabc`](https://pyabc.readthedocs.io), [`pyabcranger`](https://github.com/diyabc/abcranger), as well as an experimental integration with [`elfi`](https://github.com/diyabc/abcranger) are provided.


## Metrics

In order to compare algorithms on the benchmarks, a number of different metrics can be computed. Each task comes with reference samples for each observation. Depending on the benchmark, these are either obtained by making use of an analytic solution for the posterior or a customized likelihood-based approach.

A number of metrics can be computed by comparing algorithm samples to reference samples. In order to do so, a number of different two-sample tests can be computed (see `sbibm/metrics`). These test follow a simple interface, just requiring to pass samples from reference and algorithm.

For example, in order to compute C2ST:
```python
import torch
from sbibm.metrics.c2st import c2st
from sbibm.algorithms.mcabc import run as run_rej_abc

reference_samples = task.get_reference_posterior_samples(num_observation=1)
algorithm_samples = run_rej_abc(task=task, num_samples=10_000, num_simulation=100_000, num_observation=1)
c2st_accuracy = c2st(reference_samples, algorithm_samples)
```

For more info, see `help(c2st)`.


## Experiments

As mentioned above, we host the code for reproducing the experiments of the manuscript in a seperate repository at [github.com/sbi-benchmark/benchmarking_sbi](https://github.com/sbi-benchmark/benchmarking_sbi). Besides the pipeline to reproduce the manuscripts' experiments, full results including dataframes for quick comparisons are provided.



## License

MIT
