import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi.utils.plot import pairplot
from sklearn.utils import check_random_state


def sample_blobs_same(
    n: int, rows: int = 3, cols: int = 3, sep: int = 1, rs: int = None
) -> (torch.Tensor, torch.Tensor):
    """Generate same blobs for testing type-I error

    Source: https://github.com/fengliu90/DK-for-TST
    """
    rs = check_random_state(rs)
    correlation = 0

    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)
    X = rs.multivariate_normal(mu, sigma, size=n)
    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)

    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n) * sep
    X[:, 1] += rs.randint(cols, size=n) * sep
    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep

    return (
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(Y.astype(np.float32)),
    )


def sample_blobs_different(
    n: int,
    rows: int = 3,
    cols: int = 3,
    rs: int = None,
) -> (torch.Tensor, torch.Tensor):
    """Generate different blobs for testing type-II error (test power)

    Source: https://github.com/fengliu90/DK-for-TST
    """
    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=n)
    Y = rs.multivariate_normal(mu, np.eye(2), size=n)

    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n)
    X[:, 1] += rs.randint(cols, size=n)
    Y_row = rs.randint(rows, size=n)
    Y_col = rs.randint(cols, size=n)
    locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    # sigma matrix
    sigma_mx_2 = np.zeros([9, 2, 2])
    sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
    for i in range(9):
        sigma_mx_2[i] = sigma_mx_2_standard
        if i < 4:
            sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
            sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
        if i == 4:
            sigma_mx_2[i][0, 1] = 0.00
            sigma_mx_2[i][1, 0] = 0.00
        if i > 4:
            sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
            sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)

    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y, L) + locs[i], Y)

    return (
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(Y.astype(np.float32)),
    )


def plot_blobs_same():
    X, Y = sample_blobs_same(500, sep=10)
    pairplot([X.numpy(), Y.numpy()])
    plt.show()


def plot_blobs_different():
    X, Y = sample_blobs_different(500)
    pairplot([X.numpy(), Y.numpy()])
    plt.show()
