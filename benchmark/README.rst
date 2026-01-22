Simulation-based Inference Benchmark
====================================
|Build Status| |Python 3.10+|

Benchopt is a package to simplify and make more transparent and reproducible the comparisons of optimization algorithms. This benchmark is dedicated to simulation-based inference (SBI) algorithms. The goal of SBI is to approximate the posterior distribution of a stochastic model (or simulator):

.. math:: q_{\phi}(\theta \mid x) \approx p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)}

where :math:`\theta` denotes the model parameters and :math:`x` is an observation. In SBI the likelihood :math:`p(x \mid \theta)` is implicitly modeled by the stochastic simulator. Placing a prior :math:`p(\theta)` over the simulator parameters, allows us to generate samples from the joint distribution :math:`p(\theta, x) = p(x \mid \theta) p(\theta)` which can then be used to approximate the posterior distribution :math:`p(\theta \mid x)`, e.g. via the training of a deep generative model :math:`q_{\phi}(\theta \mid x)`.

In this benchmark we only consider amortized SBI algorithms that allow for inference for any new observation :math:`x`, without simulating new data after the initial training phase.


Environment
------------

CPU, Python 3.8 - 3.11

If a MacOS device with a M1 (ARM) processor is used, run the following before proceeding to the below installation instructions of benchopt:

.. code-block::

   conda install pyarrow


Installation
------------

This benchmark can be run using the following commands:

.. code-block::

   pip install -U benchopt
   git clone https://github.com/sbi-dev/sbibm
   cd benchmark_sbi/benchmark
   benchopt install .
   benchopt run .

Alternatively, options can be passed to ``benchopt <install/run>`` to restrict the installations/runs to some solvers or datasets:

.. code-block::

	benchopt <install/run> -s 'npe_sbi[flow=nsf]' -d 'slcp[train_size=4096'] --n-repetitions 3

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/stable/user_guide/CLI_ref.html.


Results
-------

Results are saved in the `outputs/` folder, with a `.html` file that offers a visual interface showing convergence plots for the different datasets, solvers and metrics. They were obtained by running

.. code-block::

	benchopt run --n-repetitions 10 --max-runs 1000 --timeout 1000000000000000000

where the parameters ``max-runs`` and ``timeout`` are given high values to avoid premature stopping of the algorithms without convergence.


Contributing
------------

Everyone is welcome to contribute by adding datasets, solvers (algorithms) or metrics.

* To add a dataset, add a file in the ``datasets`` folder.

  The dataset should provide training and reference test pairs of parameters and observations, as well as sbibm.Task object to get access to the prior.

* To add a solver, add a file in the ``solvers`` folder.

  Solvers represent different amortized SBI algorithms (NRE, NPE, FMPE, ...). They are initialized (``Solver.set_objective``) with the training pairs ``thetas, xs`` and the task. After training (``Solver.run``), they are expected to return (``Solver.get_result``) a function ``sample`` that generate parameters :math:`\theta \sim q_{\phi}(\theta \mid x)`.

* Metrics evaluate the quality of the estimated posterior obtained from the solver. The main metric is the C2ST metric. To add a new metric, modify the ``evaluate_result`` method of the ``Objective`` class in the ``objective.py`` file.

.. |Build Status| image:: https://github.com/JuliaLinhart/benchmark_sbi/workflows/Tests/badge.svg
   :target: https://github.com/JuliaLinhart/benchmark_sbi/actions
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
