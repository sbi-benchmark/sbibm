import numpy as np
import pytest
import torch
from pyro import distributions as pdist
from pyro import util as putil
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    IndependentTransform,
    SigmoidTransform,
)

import sbibm
from sbibm.algorithms import rej_abc
from sbibm.algorithms.sbi.snpe import run
from sbibm.metrics import c2st
from sbibm.metrics.ppc import median_distance
from sbibm.tasks.noref_beam.task import (
    NorefBeam,
    base_coordinate_field,
    bcast_coordinate_field,
    torch_average,
)

putil.set_rng_seed(47)


################################################
## API tests that related the internal task code


def test_task_constructs():

    t = NorefBeam()

    assert t


def test_multivariate_normal_constructs():

    m = torch.ones((2,))
    S = torch.eye(2)

    data_dist = pdist.MultivariateNormal(m.float(), S.float())

    assert data_dist

    sample = data_dist.sample()
    assert sample.shape == (2,)

    nensemble = 32
    sample = data_dist.sample((nensemble,))
    assert sample.shape == (nensemble, 2)


def test_multivariate_normal_shapes():

    m = torch.ones((2,))
    S = torch.eye(2)

    data_dist = pdist.MultivariateNormal(m.float(), S.float())

    event_dim = data_dist.event_dim
    assert event_dim > 0

    totshape = data_dist.shape()
    assert len(totshape) > 0
    assert totshape[0] > 0
    assert totshape[0] > event_dim

    sample = data_dist.sample()
    assert sample.shape == (2,)
    assert sample.shape == totshape


@pytest.fixture
def batched_mvn():

    batch_size = 8
    # m = torch.ones((2,))
    # m_ = torch.broadcast_to(m, (batch_size, 2))
    m_ = torch.arange(0, 2 * batch_size).reshape((batch_size, 2)).float()
    S = torch.eye(2)
    S_ = torch.broadcast_to(S, (batch_size, 2, 2))

    value = pdist.MultivariateNormal(m_.float(), S_.float())
    return value


def test_multivariate_normal_constructs_asbatch(batched_mvn):

    data_dist = batched_mvn
    assert data_dist
    batch_size = data_dist.loc.shape[0]
    assert batch_size == 8

    sample = data_dist.sample()
    assert sample.shape == (batch_size, 2)

    nensemble = 32
    sample = data_dist.sample((nensemble,))
    assert sample.shape == (nensemble, batch_size, 2)


def test_multivariate_normal_constructs_asbatch_onrange(batched_mvn):

    data_dist = batched_mvn
    batch_size = data_dist.loc.shape[0]
    assert batch_size == data_dist.batch_shape[0]
    assert data_dist.batch_shape[1:] == ()
    assert data_dist.event_shape == (2,)

    m_ = data_dist.loc

    ## single batch sample
    sample = data_dist.sample()
    assert sample.shape == (batch_size, 2)
    lp = data_dist.log_prob(sample)
    assert len(lp.shape) > 0
    assert lp.shape == data_dist.batch_shape

    ## multiple batches sample
    nensemble = 1024
    # http://pyro.ai/examples/tensor_shapes.html#Examples
    samples = data_dist.sample((nensemble,))
    assert samples.shape == (nensemble, batch_size, 2)

    m_hat = samples.mean(axis=0)
    assert torch.allclose(m_hat[0, :], m_[0, :], atol=75e-2)
    assert torch.allclose(m_hat, m_, atol=2e-1)

    lp = data_dist.log_prob(samples)
    assert len(lp.shape) > 0
    assert lp.shape == (nensemble, *data_dist.batch_shape)


def test_multivariate_normal_logprob_onfield(batched_mvn):

    data_dist = batched_mvn
    batch_size = data_dist.loc.shape[0]
    m_ = data_dist.loc
    assert data_dist

    dim_size = 100
    field = base_coordinate_field(0, dim_size)
    bfield = bcast_coordinate_field(field, batch_size, swap_to_front=False)

    with pytest.raises(ValueError) as ve:
        # this fails, because of
        # ValueError: Value is not broadcastable with batch_shape+event_shape: torch.Size([8, 100, 100, 2]) vs torch.Size([8, 2]).
        elp = torch.exp(data_dist.log_prob(bfield))

    bfield = bcast_coordinate_field(field, batch_size)
    assert bfield.shape == (dim_size, dim_size, batch_size, 2)
    #                          same as m_.shape ^^^^^^^^^^  ^
    assert bfield.shape[-2:] == m_.shape

    elp = torch.exp(data_dist.log_prob(bfield))
    assert elp.shape[0] != batch_size
    assert elp.shape == (dim_size, dim_size, batch_size)

    img = torch.moveaxis(elp, -1, 0)
    assert img.shape == (batch_size, dim_size, dim_size)

    first = img.sum(axis=1)
    second = img.sum(axis=2)

    assert first.shape == second.shape
    assert first.shape == (batch_size, dim_size)

    first_amax = torch.argmax(first, axis=-1)
    assert first_amax.shape == (batch_size,)
    assert torch.allclose(
        first_amax.float(), m_[:, 0]
    )  # we projected along y onto x (mean in x, i.e. dim at 1)

    second_amax = torch.argmax(second, axis=-1)
    assert second_amax.shape == (batch_size,)
    assert torch.allclose(
        second_amax.float(), m_[:, 1]
    )  # we projected along x onto y (mean in y, i.e. dim at 0)


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


def test_base_coordinate_field():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis

    arr = base_coordinate_field(min_axis, max_axis)

    assert arr.shape == (size_axis, size_axis, 2)
    assert np.allclose(arr[0, 0, :], np.ones(2) * min_axis)
    assert np.allclose(arr[-1, -1, :], np.ones(2) * (max_axis - 1))


def test_base_coordinate_field_halfstep():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis
    nsteps = int(size_axis / 0.5)

    arr = base_coordinate_field(min_axis, max_axis, 0.5)

    assert arr.shape == (nsteps, nsteps, 2)
    assert np.allclose(arr[0, 0, :], np.ones(2) * min_axis)
    assert np.allclose(arr[-1, -1, :], np.ones(2) * (max_axis - 0.5))


def test_base_coordinate_field_doublestep():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis
    nsteps = int(size_axis / 2.0)

    arr = base_coordinate_field(min_axis, max_axis, 2.0)

    assert arr.shape == (nsteps, nsteps, 2)
    assert np.allclose(arr[0, 0, :], np.ones(2) * min_axis)
    assert np.allclose(arr[-1, -1, :], np.ones(2) * (max_axis - 2.0))


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


def test_binomial_api_on_batched_images():

    img = torch.tensor(
        [
            [[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]],
            [[0.05, 0.2, 0.05], [0.0, 0.4, 0.0], [0.05, 0.2, 0.05]],
        ]
    )

    assert img.shape == (2, 3, 3)
    assert img[0, ...].sum() == 1.0
    assert np.isclose(img.sum().item(), 1.0 * img.shape[0])

    bdist = pdist.Binomial(total_count=1024, probs=img)
    samples = bdist.sample()
    assert samples.shape == img.shape
    assert samples.max() < 1024 * 0.5

    img_ = torch.swapaxes(img, 2, 0)
    bdist = pdist.Binomial(total_count=1024, probs=img_)
    samples = bdist.sample()
    assert samples.shape == img_.shape


def test_multivariate_normal_sample_binomial_from_logprob():

    batch_size = 8
    max_axis = batch_size * 2
    min_axis = -max_axis
    size_axis = max_axis - min_axis
    m_ = torch.arange(-batch_size, batch_size).reshape((batch_size, 2)).float()

    S = torch.eye(2)
    S_ = torch.broadcast_to(S, (batch_size, 2, 2))

    data_dist = pdist.MultivariateNormal(m_.float(), S_.float(), validate_args=False)

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


def test_bcast_coordinate_field():

    m_ = torch.arange(12).reshape(4, 3).float()
    assert m_.shape == (4, 3)
    arr = bcast_coordinate_field(m_, 2)

    assert arr.shape == (3, 4, 2)


def test_bcast_coordinate_field_swaptofront():

    m_ = torch.arange(12).reshape(4, 3).float()
    assert m_.shape == (4, 3)
    nbatches = 8

    arr = bcast_coordinate_field(m_, nbatches)
    assert arr.shape == (3, 4, nbatches)

    arr = bcast_coordinate_field(m_, nbatches, swap_to_front=False)
    assert arr.shape == (nbatches, 4, 3)


def test_simulation_output():

    t = NorefBeam()

    simulator = t.get_simulator()

    assert simulator is not None

    params = torch.tensor(
        [
            [75.6423, 26.1341, 7.7327, 10.0449],
            [66.01, 50.02, 15.03, 20.04],
            [44.05, 70.06, 25.07, 2.08],
        ]
    )

    batch_size = params.shape[0]
    assert params.shape == (batch_size, 4)

    x_t = simulator(params)

    assert x_t.shape == (batch_size, 400)

    lims = torch.arange(t.min_axis, t.max_axis, t.step_width).float()
    nsteps = t.nsteps
    assert nsteps == 200
    assert lims.shape == (nsteps,)

    for s in range(batch_size):
        left_ = x_t[s, :nsteps]
        right_ = x_t[s, nsteps:]

        left_argx = torch.argmax(left_)
        assert left_argx < 2.5 * params[s, 0]
        assert left_argx > 1.5 * params[s, 0]

        right_argx = torch.argmax(right_)
        assert right_argx < 2.5 * params[s, 1]
        assert right_argx > 1.5 * params[s, 1]

        left_m = torch_average(lims, weights=left_)
        assert left_m < 1.1 * params[s, 0]
        assert left_m > 0.9 * params[s, 0]

        right_m = torch_average(lims, weights=right_)
        assert right_m < 1.1 * params[s, 1]
        assert right_m > 0.9 * params[s, 1]


def test_pyro_batching():

    d1 = pdist.Bernoulli(0.5)

    assert d1.batch_shape == ()

    d1_ = pdist.Bernoulli(0.5 * torch.ones(3))
    assert d1_.batch_shape == (3,)

    d2 = pdist.Bernoulli(0.1 * torch.ones(3, 4))
    assert d2.batch_shape == (3, 4)

    d2_ = pdist.Bernoulli(torch.tensor([0.1, 0.2, 0.3, 0.4])).expand([3, 4])
    assert d2_.batch_shape == (3, 4)


def test_automated_transforms_to_unbounded_space():

    t = NorefBeam()

    assert hasattr(t, "_get_transforms")

    p = t.get_prior_dist()
    assert p.support
    print(type(p), type(p.support), p.support)

    obs = t._get_transforms()

    assert obs is not None
    assert "parameters" in obs.keys()

    itrf = obs["parameters"]
    roundtrf = itrf.inv

    ps = p.sample((100,))
    us = itrf(ps)
    ps_ = roundtrf(us)

    assert torch.allclose(ps, ps_)
    assert not torch.allclose(ps, us)
    print(ps.min(), ps.median(), ps.mean(), ps.std(), ps.max())
    print(us.min(), us.median(), us.mean(), us.std(), us.max())
