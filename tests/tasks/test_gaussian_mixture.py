import sbibm


def test_gaussian_mixture():
    task = sbibm.get_task("gaussian_mixture")

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    num_simulations = 100000
    theta = prior.sample((1,))
    x = simulator(theta.repeat(num_simulations, 1))

    assert x.shape == (num_simulations, 2)

    # Make sure samples do not come from just a single component.
    assert 0.6 < x.std(dim=0)[0] < 0.8, "Samples do not come from both components."
