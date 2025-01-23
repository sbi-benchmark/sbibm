import pytest

import sbibm
from sbibm.algorithms.sbi import snle, snpe, snre

@pytest.mark.parametrize("run_method", (snle, snpe, snre))
@pytest.mark.parametrize("num_rounds", (2,))
@pytest.mark.parametrize("task_name", ("gaussian_mixture",))
@pytest.mark.parametrize("num_observation", (1,))
def test_sbi_api(
    run_method: str,
    num_rounds: int,
    task_name: str,
    num_observation: int,
    num_simulations: int=2_000,
    num_samples: int=100,
):
    task = sbibm.get_task(task_name)

    kwargs = dict(
        num_rounds=num_rounds,
        training_batch_size=100,
        neural_net="mlp" if run_method == snre else "maf",
    )
    if run_method in (snle, snre):
        kwargs["mcmc_parameters"] = dict(
            num_chains=100, warmup_steps=100, thin=10, init_strategy="resample"
        )

    predicted, _, _ = run_method(
        task=task,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_samples=num_samples,
        **kwargs,
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    expected = reference_samples[:num_samples, :]

    assert expected.shape == predicted.shape
