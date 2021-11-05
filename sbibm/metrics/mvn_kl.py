import torch


def mvn_kl_pq(
    X: torch.Tensor,
    Y: torch.Tensor,
    z_score: bool = True,
) -> torch.Tensor:
    """KL(p||q) between Multivariate Normal distributions

    X and Y are both sets of samples, the mean and covariance of which is estimated
    in order to analytically calculate the KL divergence
    """

    _, num_parameters = X.shape
    if z_score:
        X_mean = torch.mean(X, axis=0)
        X_std = torch.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    try:
        X_mean = torch.mean(X, axis=0)
        X_cov = cov(X.T)

        Y_mean = torch.mean(Y, axis=0)
        Y_cov = cov(Y.T)

        # 1D tasks need cov reshape to be used in MVN below.
        if num_parameters == 1:
            X_cov = X_cov.reshape(1, 1)
            Y_cov = Y_cov.reshape(1, 1)

        p = torch.distributions.MultivariateNormal(loc=X_mean, covariance_matrix=X_cov)
        q = torch.distributions.MultivariateNormal(loc=Y_mean, covariance_matrix=Y_cov)
        kl = torch.distributions.kl_divergence(p, q)
    except RuntimeError:
        kl = torch.tensor(float("nan"))

    return kl


def mvn_kl_qp(X: torch.Tensor, Y: torch.Tensor, z_score: bool = True) -> torch.Tensor:
    """KL(q||p) between Multivariate Normal distributions

    X and Y are both sets of samples, the mean and covariance of which is estimated
    in order to analytically calculate the KL divergence
    """
    return mvn_kl_pq(Y, X, z_score=z_score)


def cov(m, rowvar=True, inplace=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.

    Note:
        https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
