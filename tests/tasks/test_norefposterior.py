import numpy as np
import pytest
import torch
from pyro import distributions as pdist

import sbibm
from sbibm.tasks.norefposterior.task import (
    norefposterior,
    quadratic_coordinate_field,
    torch_average,
)

########### sbibm related ################
## testing the actual task


def test_task_constructs():

    t = norefposterior()

    assert t


def test_obtain_task():

    task = sbibm.get_task("norefposterior")

    assert task is not None


def test_obtain_prior():

    task = sbibm.get_task(
        "norefposterior"
    )  # See sbibm.get_available_tasks() for all tasks
    prior = task.get_prior()

    assert prior is not None


def test_obtain_simulator():

    task = sbibm.get_task("norefposterior")

    simulator = task.get_simulator()

    assert simulator is not None


def test_obtain_observe_once():

    task = sbibm.get_task("norefposterior")

    x_o = task.get_observation(num_observation=1)

    assert x_o is not None
    assert hasattr(x_o, "shape")


def test_obtain_prior_samples():

    task = sbibm.get_task("norefposterior")
    prior = task.get_prior()
    nsamples = 10

    thetas = prior(num_samples=nsamples)

    assert thetas.shape == (nsamples, 4)


def test_simulate_from_thetas():

    task = sbibm.get_task("norefposterior")
    prior = task.get_prior()
    sim = task.get_simulator()
    nsamples = 10

    thetas = prior(num_samples=nsamples)
    xs = sim(thetas)

    assert xs.shape == (nsamples, 400)


################################################
## sbibm compliant API tests as documented in
## the top-level README.md

@pytest.fixture
def vanilla_samples():

    task = sbibm.get_task("norefposterior")
    prior = task.get_prior()
    sim = task.get_simulator()
    nsamples = 10

    thetas = prior(num_samples=nsamples)
    xs = sim(thetas)

    return task, thetas, xs


def test_quick_demo_rej_abc(vanilla_samples):

    task, thetas, xs = vanilla_samples
    from sbibm.algorithms import rej_abc
    posterior_samples, _, _ = rej_abc(task=task,
                                      num_samples=50,
                                      num_observation=1,
                                      num_simulations=500)

    assert posterior_samples != None


def test_quick_demo_c2st(vanilla_samples):

    task, thetas, xs = vanilla_samples
    from sbibm.algorithms import rej_abc
    posterior_samples, _, _ = rej_abc(task=task,
                                      num_samples=50,
                                      num_observation=1,
                                      num_simulations=500)

    from sbibm.metrics import c2st
    reference_samples = task.get_reference_posterior_samples(num_observation=1)
    c2st_accuracy = c2st(reference_samples, posterior_samples)

    assert c2st_accuracy > 0.
    assert c2st_accuracy < 1.


################################################
## API tests that related the internal task code


def test_multivariate_normal_constructs():

    m = torch.ones((2,))
    S = torch.eye(2)

    data_dist = pdist.MultivariateNormal(m.float(), S.float())

    assert data_dist

    sample = data_dist.sample()
    assert sample.shape == (2,)

    nensemble = 32
    sample = data_dist.sample_n(nensemble)
    assert sample.shape == (nensemble, 2)


def test_multivariate_normal_constructs_asbatch():

    batch_size = 8
    m = torch.ones((2,))
    m_ = torch.broadcast_to(m, (batch_size, 2))
    S = torch.eye(2)
    S_ = torch.broadcast_to(S, (batch_size, 2, 2))

    data_dist = pdist.MultivariateNormal(m_.float(), S_.float())

    assert data_dist

    sample = data_dist.sample()
    assert sample.shape == (batch_size, 2)

    nensemble = 32
    sample = data_dist.sample((nensemble,))
    assert sample.shape == (nensemble, batch_size, 2)


def test_multivariate_normal_constructs_asbatch_onrange():

    batch_size = 8
    m_ = torch.arange(0, 2 * batch_size).reshape((batch_size, 2)).float()

    S = torch.eye(2)
    S_ = torch.broadcast_to(S, (batch_size, 2, 2))

    data_dist = pdist.MultivariateNormal(m_.float(), S_.float())

    assert data_dist

    sample = data_dist.sample()
    assert sample.shape == (batch_size, 2)

    nensemble = 1024
    samples = data_dist.sample((nensemble,))
    assert samples.shape == (nensemble, batch_size, 2)

    m_hat = samples.mean(axis=0)

    assert torch.allclose(m_hat[0, :], m_[0, :], atol=75e-2)
    assert torch.allclose(m_hat, m_, atol=2e-1)


def test_prepare_coordinates():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis

    # prepare two grids for x and y
    x = torch.arange(min_axis, max_axis).detach().float()
    y = torch.arange(min_axis, max_axis).detach().float()

    xx, yy = torch.meshgrid(x, y)
    val = torch.swapaxes(torch.stack((xx.flatten(), yy.flatten())), 1, 0).float()

    valr = val.reshape(size_axis, size_axis, 2)

    # at every point of the image w=size_axis x w=size_axis
    # we store the (x,y) coordinate of a regular grid
    # so we get:
    # valr[0,0] = (-16,-16),
    # valr[0,1] = (-16,-15),
    # valr[0,2] = (-16,-14)

    assert valr.shape == (size_axis, size_axis, 2)
    assert not torch.allclose(valr[0, 0], valr[0, 1])
    assert torch.allclose(valr[0, 0], valr[1, 1] - 1.0)
    assert valr[-1, 0, 0] == valr[0, -1, 1]

    # broadcast to <batchsize> doublicates
    valr_ = torch.broadcast_to(valr, (batch_size, *valr.shape)).detach()

    # move axis from position 2 to front
    valb = torch.swapaxes(valr_, 2, 0)

    # we store the x and y coordinates as 2 images
    assert valb.shape == (size_axis, size_axis, batch_size, 2)
    assert torch.allclose(valb[:, :, 0, :], valb[:, :, 1, :])


def test_quadratic_coordinate_field():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis

    arr = quadratic_coordinate_field(min_axis, max_axis, batch_size)

    assert arr.shape == (size_axis, size_axis, batch_size, 2)
    assert torch.allclose(arr[:, :, 0, :], arr[:, :, 1, :])
    assert torch.allclose(arr[:, :, 0, :], arr[:, :, -1, :])
    assert torch.allclose(arr[:, :, batch_size // 2, :], arr[:, :, -1, :])


def test_binomial_api():

    img = torch.tensor([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])

    assert img.sum() == 1.0
    bdist = pdist.Binomial(total_count=1024, probs=img)
    samples = bdist.sample()
    assert samples.shape == img.shape
    assert samples.max() < 1024 * 0.5

    lims = np.arange(3)
    mean = np.average(lims, weights=samples.sum(axis=0).numpy(), axis=0)
    assert np.allclose(mean, 1, atol=1e-1)


def test_multivariate_normal_sample_binomial_from_logprob():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis
    m_ = torch.arange(-batch_size, batch_size).reshape((batch_size, 2)).float()

    S = torch.eye(2)
    S_ = torch.broadcast_to(S, (batch_size, 2, 2))

    data_dist = pdist.MultivariateNormal(m_.float(), S_.float())

    x = torch.arange(min_axis, max_axis).detach().float()
    y = torch.arange(min_axis, max_axis).detach().float()
    xx, yy = torch.meshgrid(x, y)
    val = torch.swapaxes(torch.stack((xx.flatten(), yy.flatten())), 1, 0).float()
    valr = val.reshape(size_axis, size_axis, 2)

    valr_ = torch.broadcast_to(valr, (batch_size, *valr.shape)).detach()
    valb = torch.swapaxes(valr_, 2, 0)

    # TODO: replace this with sampling
    # create images from probabilities
    img = torch.exp(data_dist.log_prob(valb))
    assert img.shape == (size_axis, size_axis, batch_size)

    # sample this using a binomial
    bdist = pdist.Binomial(total_count=1024 * 16, probs=img)
    samples = bdist.sample()

    # shapes are correct of the sampled image
    assert samples.shape == valb.shape[:-1]
    assert samples.shape == img.shape

    samples_tox = torch.sum(samples, axis=0)
    samples_toy = torch.sum(samples, axis=1)

    # shapes of projections to specific axes are correct
    assert samples_tox.shape == samples_toy.shape
    assert samples_tox.shape == (size_axis, batch_size)

    x_ = torch.broadcast_to(x, (batch_size, size_axis))
    xt = torch.swapaxes(x_, 1, 0)

    assert xt.shape == samples_tox.shape

    # compare mean values per axis with the originals
    # defined at the beginning of this function
    m_hat0 = torch.sum(xt * samples_tox, axis=0) / torch.sum(samples_tox, axis=0)
    assert m_hat0.shape == (batch_size,)
    assert torch.allclose(m_hat0, m_[:, 0], atol=1e-1)

    m_hat1 = torch.sum(xt * samples_toy, axis=0) / torch.sum(samples_toy, axis=0)
    assert m_hat1.shape == (batch_size,)
    assert torch.allclose(m_hat1, m_[:, 1], atol=1e-1)


def test_torch_average():

    m_ = 5 * torch.arange(1, 3).float()
    S = torch.eye(2).float()

    data_dist = pdist.MultivariateNormal(m_, S)

    samples = data_dist.sample((2048,))

    bins0, edges0 = np.histogram(samples[:, 0].numpy(), bins=15)

    m0 = torch_average(torch.from_numpy(edges0[:-1]), torch.from_numpy(bins0))

    assert m0 > 4.0
    assert m0 < 6.0

    bins1, edges1 = np.histogram(samples[:, 1].numpy(), bins=15)

    m1 = torch_average(torch.from_numpy(edges1[:-1]), torch.from_numpy(bins1))

    assert m1 > 9.0
    assert m1 < 11.0
