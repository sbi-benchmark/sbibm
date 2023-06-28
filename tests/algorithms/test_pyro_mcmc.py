import sbibm

from sbibm.algorithms.pyro.mcmc import run


def test_pyro_mcmc():
    task = sbibm.get_task("gaussian_linear")

    samples = run(
        task, num_chains=1, num_samples=100, num_warmup=100, num_observation=1
    )
