import re

import pyro
import pytest
import torch

from sbibm import get_available_tasks, get_task

pyro.util.set_rng_seed(47)

all_tasks = set(get_available_tasks())
julia_tasks = set([tn for tn in get_available_tasks() if re.search("lotka|sir", tn)])
noref_tasks = set([tn for tn in get_available_tasks() if "noref" in tn])


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_task_can_be_obtained(task_name):

    task = get_task(task_name)

    assert task is not None


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_obtain_prior_from_task(task_name):

    task = get_task(task_name)
    prior = task.get_prior()

    assert prior is not None


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_obtain_simulator_from_task(task_name):

    task = get_task(task_name)

    simulator = task.get_simulator()

    assert simulator is not None


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_retrieve_observation_from_task(task_name):

    task = get_task(task_name)

    x_o = task.get_observation(num_observation=1)

    assert x_o is not None
    assert hasattr(x_o, "shape")
    assert len(x_o.shape) > 1


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_describe_theta(task_name):

    task = get_task(task_name)

    labels = task.get_labels_parameters()

    assert isinstance(labels, list)
    assert len(labels) == task.get_true_parameters(num_observation=1).shape[-1]


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_describe_x(task_name):

    task = get_task(task_name)

    labels = task.get_labels_data()

    assert isinstance(labels, list)
    assert len(labels) == task.get_observation(num_observation=1).shape[-1]


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_obtain_prior_samples_from_task(task_name):

    task = get_task(task_name)
    prior = task.get_prior()
    nsamples = 10

    thetas = prior(num_samples=nsamples)

    assert thetas.shape[0] == nsamples


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_simulate_from_thetas(task_name):

    task = get_task(task_name)
    prior = task.get_prior()
    sim = task.get_simulator()
    nsamples = 10

    thetas = prior(num_samples=nsamples)
    xs = sim(thetas)

    assert xs.shape[0] == nsamples


@pytest.mark.parametrize(
    "task_name", [tn for tn in (all_tasks - julia_tasks - noref_tasks)]
)
def test_reference_posterior_exists(task_name):

    task = get_task(task_name)

    reference_samples = task.get_reference_posterior_samples(num_observation=1)

    assert hasattr(reference_samples, "shape")
    assert len(reference_samples.shape) == 2
    assert reference_samples.shape[0] > 0


@pytest.mark.parametrize("task_name", [tn for tn in noref_tasks])
def test_reference_posterior_not_called(task_name):

    task = get_task(task_name)

    with pytest.raises(NotImplementedError):
        reference_samples = task.get_reference_posterior_samples(num_observation=1)

    assert task is not None


@pytest.mark.parametrize("task_name", [tn for tn in (all_tasks - julia_tasks)])
def test_transforms_shapes(task_name, batch_size=5):
    task = get_task(task_name)
    prior = task.get_prior()
    samples = prior(num_samples=batch_size)

    transforms = task._get_transforms(True)["parameters"]

    ladj_shape = transforms.log_abs_det_jacobian(transforms(samples), samples).shape
    assert ladj_shape == torch.Size([batch_size])

    assert transforms is not None
