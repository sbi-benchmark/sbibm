[![PyPI
version](https://img.shields.io/pypi/v/sbibm)](https://pypi.org/project/sbibm/) ![Python versions](https://img.shields.io/pypi/pyversions/sbibm) [![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sbi-benchmark/sbibm/blob/master/CONTRIBUTING.md) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)

# Simulation-Based Inference Benchmark

This repository contains a simulation-based inference benchmark framework, `sbibm`, which we describe in the [associated manuscript "Benchmarking Simulation-based Inference"](http://proceedings.mlr.press/v130/lueckmann21a.html). A short summary of the paper and interactive results can be found on the project website: https://sbi-benchmark.github.io

The benchmark framework includes tasks, reference posteriors, metrics, plotting, and integrations with SBI toolboxes. The framework is designed to be highly extensible and easily used in new research projects as we show below.

In order to emphasize that `sbibm` can be used independently of any particular analysis pipeline, we split the code for reproducing the experiments of the manuscript into a seperate repository hosted at [github.com/sbi-benchmark/results/](https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi). Besides the pipeline to reproduce the manuscripts' experiments, full results including dataframes for quick comparisons are hosted in that repository.

If you have questions or comments, please do not hesitate [to contact us](mailto:mail@jan-matthis.de) or [open an issue](https://github.com/sbi-benchmark/sbibm/issues). We [invite contributions](CONTRIBUTING.md), e.g., of new tasks, novel metrics, or wrappers for other SBI toolboxes.


## Installation

Assuming you have a working Python environment, simply install `sbibm` via `pip`:
```commandline
$ pip install sbibm
```

ODE based models (currently SIR and Lotka-Volterra models) use [Julia](https://julialang.org) via [`diffeqtorch`](https://github.com/sbi-benchmark/diffeqtorch). If you are planning to use these tasks, please additionally follow the [installation instructions of `diffeqtorch`](https://github.com/sbi-benchmark/diffeqtorch#installation). If you are not planning to simulate these tasks for now, you can skip this step.

## Quickstart

A quick demonstration of `sbibm`, see further below for more in-depth explanations:

```python
import sbibm

task = sbibm.get_task("two_moons")  # See sbibm.get_available_tasks() for all tasks
prior = task.get_prior()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)  # 10 per task

# These objects can then be used for custom inference algorithms, e.g.
# we might want to generate simulations by sampling from prior:
thetas = prior(num_samples=10_000)
xs = simulator(thetas)

# Alternatively, we can import existing algorithms, e.g:
from sbibm.algorithms import rej_abc  # See help(rej_abc) for keywords
posterior_samples, _, _ = rej_abc(task=task, num_samples=10_000, num_observation=1, num_simulations=100_000)

# Once we got samples from an approximate posterior, compare them to the reference:
from sbibm.metrics import c2st
reference_samples = task.get_reference_posterior_samples(num_observation=1)
c2st_accuracy = c2st(reference_samples, posterior_samples)

# Visualise both posteriors:
from sbibm.visualisation import fig_posterior
fig = fig_posterior(task_name="two_moons", observation=1, samples=[posterior_samples])  
# Note: Use fig.show() or fig.save() to show or save the figure

# Get results from other algorithms for comparison:
from sbibm.visualisation import fig_metric
results_df = sbibm.get_results(dataset="main_paper.csv")
fig = fig_metric(results_df.query("task == 'two_moons'"), metric="C2ST")
```

## Tasks

You can then see the list of available tasks by calling `sbibm.get_available_tasks()`. If we wanted to use, say, the `two_moons` task, we can load it using `sbibm.get_task`, as in:

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

As mentioned in the intro, `sbibm` wraps a number of third-party packages to run various algorithms. We found it easiest to give each algorithm the same interface: In general, each algorithm specifies a `run` function that gets `task` and hyperparameters as arguments, and eventually returns the required `num_posterior_samples`. That way, one can simply import the run function of an algorithm, tune it on any given task, and return metrics on the returned samples. Wrappers for external toolboxes implementing algorithms are in the subfolder `sbibm/algorithms`. Currently, integrations with [`sbi`](https://www.mackelab.org/sbi/), [`pyabc`](https://pyabc.readthedocs.io), [`pyabcranger`](https://github.com/diyabc/abcranger), as well as an experimental integration with [`elfi`](https://github.com/elfi-dev/elfi) are provided.


## Metrics

In order to compare algorithms on the benchmarks, a number of different metrics can be computed. Each task comes with reference samples for each observation. Depending on the benchmark, these are either obtained by making use of an analytic solution for the posterior or a customized likelihood-based approach.

A number of metrics can be computed by comparing algorithm samples to reference samples. In order to do so, a number of different two-sample tests can be computed (see `sbibm/metrics`). These test follow a simple interface, just requiring to pass samples from reference and algorithm.

For example, in order to compute C2ST:
```python
import torch
from sbibm.metrics.c2st import c2st
from sbibm.algorithms import rej_abc

reference_samples = task.get_reference_posterior_samples(num_observation=1)
algorithm_samples, _, _ = rej_abc(task=task, num_samples=10_000, num_simulations=100_000, num_observation=1)
c2st_accuracy = c2st(reference_samples, algorithm_samples)
```

For more info, see `help(c2st)`.


## Figures

`sbibm` includes code for plotting results, for instance, to plot metrics on a specific task:

```python
from sbibm.visualisation import fig_metric

results_df = sbibm.get_results(dataset="main_paper.csv")
results_subset = results_df.query("task == 'two_moons'")
fig = fig_metric(results_subset, metric="C2ST")  # Use fig.show() or fig.save() to show or save the figure
```

It can also be used to plot posteriors, e.g., to compare the results of an inference algorithm against reference samples:

```python
from sbibm.visualisation import fig_posterior
fig = fig_posterior(task_name="two_moons", observation=1, samples=[algorithm_samples])
```


## Results and Experiments

We host results and the code for reproducing the experiments of the manuscript in a seperate repository at [github.com/sbi-benchmark/results](https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi): This includes the pipeline to reproduce the manuscripts' experiments as well as dataframes for new comparisons.


## Citation

The manuscript is [available through PMLR](http://proceedings.mlr.press/v130/lueckmann21a.html):

```bibtex
 @InProceedings{lueckmann2021benchmarking,
  title     = {Benchmarking Simulation-Based Inference},
  author    = {Lueckmann, Jan-Matthis and Boelts, Jan and Greenberg, David and Goncalves, Pedro and Macke, Jakob},
  booktitle = {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  pages     = {343--351},
  year      = {2021},
  editor    = {Banerjee, Arindam and Fukumizu, Kenji},
  volume    = {130},
  series    = {Proceedings of Machine Learning Research},
  month     = {13--15 Apr},
  publisher = {PMLR}
}  
```


## Support

This work was supported by the German Research Foundation (DFG; SFB 1233 PN 276693517, SFB 1089, SPP 2041, Germany’s Excellence Strategy – EXC number 2064/1 PN 390727645) and the German Federal Ministry of Education and Research (BMBF; project ’[ADIMEM](https://fit.uni-tuebingen.de/Project/Details?id=9199)’, FKZ 01IS18052 A-D).


## License

MIT
