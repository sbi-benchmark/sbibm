import re

import pyro
import pytest
import torch

from sbibm import get_available_tasks, get_task
from sbibm.algorithms import rej_abc
from sbibm.metrics import c2st

pyro.util.set_rng_seed(47)

task_list = [tn for tn in get_available_tasks() if not re.search("noref|lotka|sir", tn)]


@pytest.mark.parametrize("task_name", task_list)
def test_quick_demo_rej_abc(task_name):

    task = get_task(task_name)
    posterior_samples, _, _ = rej_abc(
        task=task, num_samples=12, num_observation=1, num_simulations=500
    )

    assert posterior_samples != None
    assert posterior_samples.shape[0] == 12


@pytest.mark.parametrize("task_name", task_list)
def test_quick_demo_c2st(task_name):

    task = get_task(task_name)
    posterior_samples, _, _ = rej_abc(
        task=task, num_samples=50, num_observation=1, num_simulations=500
    )

    reference_samples = task.get_reference_posterior_samples(num_observation=1)
    c2st_accuracy = c2st(reference_samples, posterior_samples)

    assert c2st_accuracy > 0.0
    assert c2st_accuracy < 1.0
