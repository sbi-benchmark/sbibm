import sbibm

from sbibm.algorithms.sbi.snre import run as run_snre
from sbibm.algorithms.sbi.snpe import run as run_snpe

from sbibm.metrics.c2st import c2st


def test_loading_ddm_task():
    sbibm.get_task("ddm")


def test_simulation_ddm_task():
    task = sbibm.get_task("ddm")
    prior = task.get_prior()
    simulator = task.get_simulator()
    simulator(prior(1))


def test_inference_with_nre():
    task = sbibm.get_task("ddm")
    num_observation = 1
    num_simulations = 10000
    num_samples = 1000

    samples, num_simulations, _ = run_snre(
        task,
        num_samples=num_samples,
        num_simulations=num_simulations,
        num_observation=num_observation,
        num_rounds=1,
    )

    reference_samples = task.get_reference_posterior_samples(num_observation)[
        :num_samples
    ]
    score = c2st(reference_samples, samples)
    assert score <= 0.6, f"score={score} must be below 0.6"
