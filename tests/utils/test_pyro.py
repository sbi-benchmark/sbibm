import pytest
import torch

import sbibm


@pytest.mark.parametrize(
    "task_name,jit_compile,batch_size,implementation,posterior",
    [
        (task_name, jit_compile, batch_size, implementation, posterior)
        for task_name in ["gaussian_linear"]
        for jit_compile in [False, True]
        for batch_size in [1, 2]
        for implementation in ["pyro", "experimental"]
        for posterior in [True, False]
    ],
)
def test_log_prob_fn(task_name, jit_compile, batch_size, implementation, posterior):
    """Test `get_log_prob_fn`

    Uses test cases for which the true posterior is known in closed form. Since
    `log_prob_fn` returns the unnormalized posterior log probability, it is tested
    whether the two are proportional.
    """
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    prior_dist = task.get_prior_dist()
    posterior_dist = task._get_reference_posterior(num_observation=1)

    log_prob = task._get_log_prob_fn(
        num_observation=1,
        implementation=implementation,
        jit_compile=jit_compile,
        posterior=posterior,
    )

    parameters = prior(num_samples=batch_size)

    # Test whether batching works
    if batch_size > 1:
        for b in range(batch_size):
            torch.allclose(
                log_prob(parameters)[b], log_prob(parameters[b, :].reshape(1, -1))
            )
            torch.allclose(
                posterior_dist.log_prob(parameters)[b],
                posterior_dist.log_prob(parameters[b, :].reshape(1, -1)),
            )

    # Test whether proportionality holds
    diff_ref = log_prob(parameters) - posterior_dist.log_prob(parameters)
    if not posterior:
        diff_ref += prior_dist.log_prob(parameters)
    for _ in range(10):
        parameters = prior(num_samples=batch_size)
        diff = log_prob(parameters) - posterior_dist.log_prob(parameters)
        if not posterior:
            diff += prior_dist.log_prob(parameters)
        assert torch.allclose(diff, diff_ref)


@pytest.mark.parametrize(
    "jit_compile,batch_size,implementation",
    [
        (jit_compile, batch_size, implementation)
        for jit_compile in [False, True]
        for batch_size in [1, 2]
        for implementation in ["pyro"]
    ],
)
def test_log_prob_grad_fn(jit_compile, batch_size, implementation):
    """Test `get_log_prob_grad_fn`

    We are using the likleihood of the Gaussian linear using the fact that:

        ∇ wrt p of log N(p|0, 1) is -p,

    since that is the derivative of -((p-0)**2)/(2.*1.).

    We are checking against this analytical derivative.
    """
    task = sbibm.get_task("gaussian_linear", simulator_scale=1.0)
    observation = torch.zeros((10,))
    prior = task.get_prior()

    log_prob_grad = task._get_log_prob_grad_fn(
        observation=observation,
        implementation=implementation,
        jit_compile=jit_compile,
        posterior=False,
    )

    parameters = prior(num_samples=batch_size)

    # Test whether batching works
    if batch_size > 1:
        for b in range(batch_size):
            torch.allclose(
                log_prob_grad(parameters)[b],
                log_prob_grad(parameters[b, :].reshape(1, -1)),
            )

    # Test whether gradient is correct
    grads = log_prob_grad(parameters)
    analytical_grad = -1.0 * parameters
    assert torch.allclose(grads, analytical_grad)


def test_transforms():
    task = sbibm.get_task("gaussian_linear_uniform")

    observation = task.get_observation(num_observation=1)
    true_parameters = task.get_true_parameters(num_observation=1)

    transforms = task._get_transforms(automatic_transform_enabled=True)["parameters"]
    parameters_constrained = true_parameters
    parameters_unconstrained = transforms(true_parameters)

    lpf_1 = task._get_log_prob_fn(
        observation=observation, automatic_transform_enabled=False
    )
    log_prob_1 = lpf_1(parameters_constrained)

    lpf_2 = task._get_log_prob_fn(
        observation=observation, automatic_transform_enabled=True
    )
    # lpf_2 takes unconstrained parameters are inputs and returns
    # the log prob of the unconstrained distribution
    log_prob_2 = lpf_2(parameters_unconstrained)

    # through change of variables, we can recover the original log prob
    # ladj(x,y) -> log |dy/dx| -> ladj(untransformed, transformed)
    log_prob_3 = log_prob_2 + torch.sum(
        transforms.log_abs_det_jacobian(
            parameters_constrained, parameters_unconstrained
        )
    )

    assert torch.allclose(log_prob_1, log_prob_3)
