import pickle
import tempfile
from pathlib import Path

import pytest

import sbibm
from sbibm.algorithms.sbi.snre import run as run_posterior
from sbibm.metrics.c2st import c2st


# a fast test
@pytest.mark.parametrize(
    "task_name,num_observation",
    [
        (task_name, num_observation)
        for task_name in [
            "gaussian_linear",
            "gaussian_linear_uniform",
        ]
        for num_observation in [1, 3]
    ],
)
def test_nre_posterior(
    task_name, num_observation, num_simulations=2_000, num_samples=100
):
    task = sbibm.get_task(task_name)

    predicted, _, _ = run_posterior(
        task=task,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_samples=num_samples,
        num_rounds=1,
        max_num_epochs=30,
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    expected = reference_samples[:num_samples, :]

    assert expected.shape == predicted.shape

    acc = c2st(predicted, expected)

    assert acc > 0.5
    assert acc <= 1.0
    assert acc > 0.6


@pytest.mark.parametrize("task_name", ("gaussian_linear",))
def test_stored_nre_posterior(
    task_name, num_observation=4, num_simulations=2_000, num_samples=100
):
    task = sbibm.get_task(task_name)
    th, tfile_ = tempfile.mkstemp(".pkl")
    tfile = Path(tfile_)

    predicted, _, _ = run_posterior(
        task=task,
        num_observation=num_observation,
        num_simulations=num_simulations,
        num_samples=num_samples,
        num_rounds=1,
        max_num_epochs=30,
        posterior_path=tfile_,
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    expected = reference_samples[:num_samples, :]

    assert expected.shape == predicted.shape

    acc = c2st(predicted, expected)

    assert acc > 0.5
    assert acc <= 1.0
    assert acc > 0.6

    assert tfile.exists()
    assert tfile.stat().st_size > 0

    # reload and pickle the KDEWrapper
    with tfile.open("rb") as rfile:
        rposterior = pickle.load(rfile)
        obs = rposterior.sample((num_samples,))
        assert obs.shape == expected.shape
        acc = c2st(obs, expected)
        assert acc > 0.8
        print(f"reloaded posterior samples versus expected samples, c2st score = {acc}")

    # clean up after ourselves
    tfile.unlink()
