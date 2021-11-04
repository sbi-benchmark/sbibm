import torch

from sbibm.metrics import c2st

from .utils import sample_blobs_same


def test_c2st():
    X, Y = sample_blobs_same(n=10_000)

    acc = c2st(X=X, Y=Y)

    assert torch.allclose(acc, torch.tensor([0.5]), rtol=0.01, atol=0.01)
