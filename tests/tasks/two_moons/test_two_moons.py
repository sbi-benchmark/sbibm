import pyro
import pytest
import torch

from sbibm.tasks.two_moons.task import TwoMoons

pyro.util.set_rng_seed(47)

## a test suite that can be used for task internal code


def test_task_constructs():
    """this test demonstrates how to test internal task code"""

    t = TwoMoons()

    assert t
