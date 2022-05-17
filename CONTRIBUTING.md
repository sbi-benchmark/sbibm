# Contributing

We invite contributions of new algorithms, tasks, and metrics. Please do not hestitate to get in touch [via email](mailto:mail@jan-matthis.de,jakob.macke@uni-tuebingen.de) or by opening an issue on the repository. 

## Setup

To create a reproducible environment, we advice our contributors to follow these steps in order to ensure comparable results when contributing a feature or bugfix. Assuming you have already created a branch from `main` and are currently working from the root of the repo, do:

```bash
python -m venv my-branch-venv
```

load this virtual environment, e.g. on \*nix or mac do:

```bash
source my-branch-venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip 
```

Install dependencies including extras for development:

```bash
python -m pip install -e .[dev] 
```

Run the tests (this currently takes quite a while):

```bash
python -m pytest -x . 
```

To complete, run house keeping apps and try to make all errors disappear:

```bash
black .
```

```bash
isort .
```

Now commit and push your changes to github and open a PR. Thank you!


## Tasks

Adding new tasks is straightforward. It is easiest to model them after existing tasks. First, take a close look at the base class for tasks in `sbibm/tasks/task.py`: you will find a `_setup` method: This method samples from the prior, generates observations, and finally calls `_sample_reference_posterior`, to generate samples from the reference posterior. All of these results are stored in csv files, and the generation of reference posterior samples happens in parallel.

For some tasks, e.g., the `gaussian_linear`, a closed form solution for the posterior is available, which is used in `_sample_reference_posterior`, while other tasks utilize MCMC. 

Note also that each individual tasks ends with a `if __name__ == "__main__"` block at the end which calls `_setup`. This means that `_setup` is executed by calling `python sbibm/tasks/task_name/task.py`. This step overrides the existing reference posterior data, which is in the subfolder `sbibm/tasks/task_name/files/`. It should only be executed whenever a task is changed (and never by a user).


## Algorithms

To add new algorithms, take a look at the inferfaces to other third-party packages inside `sbibm/algorithms`. In general, each algorithm specifies a `run` function that gets `task` and hyperparameters as arguments, and eventually returns the required `num_posterior_samples`. Using the task instance, all task-relevant functions and settings can be obtained. We are glad to help with implementing new algorithm wrappers or adjusting exiting ones.


## Code style

For docstrings and comments, we use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). We use automatic code formatters (use `pip install sbibm[dev]` to install them). In particular:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can run black manually from the console using `black .` in the top directory of the repository, which will format all files.

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order imports. You can run isort manually from the console using `isort .` in the top directory.
