import sbibm
import torch

from sbibm.metrics.c2st import c2st

from sbi.inference import SNRE, SNPE, MNLE

from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.utils import posterior_nn


mcmc_parameters = dict(
    num_chains=50,
    thin=10,
    warmup_steps=50,
    init_strategy="proposal",
)


def test_loading_ddm_task():
    sbibm.get_task("ddm")


def test_simulation_ddm_task():
    task = sbibm.get_task("ddm")
    prior = task.get_prior()
    simulator = task.get_simulator()
    simulator(prior(1))


def map_x_to_two_D(x):
    x = x.squeeze()
    x_2d = torch.zeros(x.shape[0], 2)
    x_2d[:, 0] = x.abs()
    x_2d[x >= 0, 1] = 1

    return x_2d


def test_inference_with_nre():
    task = sbibm.get_task("ddm")
    num_observation = 101
    num_simulations = 10000
    num_samples = 1000
    x_o = map_x_to_two_D(task.get_observation(num_observation))

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    theta = prior.sample((num_simulations,))
    x = map_x_to_two_D(simulator(theta))

    trainer = SNRE(prior)
    trainer.append_simulations(theta, x).train()
    posterior = trainer.build_posterior(
        mcmc_method="slice_np_vectorized", mcmc_parameters=mcmc_parameters
    )
    samples = posterior.sample((num_samples,), x=x_o)

    reference_samples = task.get_reference_posterior_samples(num_observation)[
        :num_samples
    ]
    score = c2st(reference_samples, samples)
    print(score)
    assert score <= 0.6, f"score={score} must be below 0.6"


def test_inference_with_mnle():
    task = sbibm.get_task("ddm")
    num_observation = 101
    num_simulations = 10000
    num_samples = 1000
    x_o = map_x_to_two_D(task.get_observation(num_observation))

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    theta = prior.sample((num_simulations,))
    x = map_x_to_two_D(simulator(theta))

    trainer = MNLE(prior)
    trainer.append_simulations(theta, x).train()
    posterior = trainer.build_posterior(
        mcmc_method="slice_np_vectorized", mcmc_parameters=mcmc_parameters
    )
    samples = posterior.sample((num_samples,), x=x_o)

    reference_samples = task.get_reference_posterior_samples(num_observation)[
        :num_samples
    ]
    score = c2st(reference_samples, samples)
    print(score)
    assert score <= 0.6, f"score={score} must be below 0.6"


def test_inference_with_npe():
    task = sbibm.get_task("ddm")
    num_observation = 101
    num_simulations = 10000
    num_samples = 1000
    x_o = map_x_to_two_D(task.get_observation(num_observation))
    num_trials = x_o.shape[0]

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    theta = prior.sample((num_simulations,))

    # copy theta for iid trials
    theta_per_trial = theta.tile(num_trials).reshape(num_simulations * num_trials, -1)
    x = map_x_to_two_D(simulator(theta_per_trial))

    # rearrange to have trials as separate dim
    x = x.reshape(num_simulations, num_trials, 2)

    single_trial_net = FCEmbedding(
        input_dim=2,
        output_dim=4,
        num_hiddens=10,
        num_layers=2,
    )

    embedding_net = PermutationInvariantEmbedding(
        trial_net=single_trial_net,
        trial_net_output_dim=4,
        combining_operation="mean",
        num_layers=2,
        num_hiddens=20,
        output_dim=10,
    )

    de_provider = posterior_nn(
        model="mdn", num_components=4, embedding_net=embedding_net
    )

    trainer = SNPE(prior, density_estimator=de_provider).append_simulations(theta, x)
    trainer.train()
    posterior = trainer.build_posterior()
    samples = posterior.sample((num_samples,), x=x_o)

    reference_samples = task.get_reference_posterior_samples(num_observation)[
        :num_samples
    ]
    score = c2st(reference_samples, samples)
    print(score)
    assert score <= 0.6, f"score={score} must be below 0.6"
