import torch

from sbibm.metrics import mmd

from .utils import sample_blobs_same


def test_mmd():
    X, Y = sample_blobs_same(n=1000)

    mmd_1 = mmd(X=X, Y=Y, implementation="tp_sutherland")
    mmd_2 = mmd(X=X, Y=Y, implementation="tp_djolonga")

    assert torch.allclose(mmd_1, mmd_2, rtol=1e-04, atol=1e-04)
