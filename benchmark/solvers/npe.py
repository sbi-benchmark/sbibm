r"""Solver module for NPE, :mod:`sbi` implementation.

"""

from benchopt import BaseSolver

from sbi.inference import NPE


class Solver(BaseSolver):
    r"""Neural posterior estimation (NPE).

    The solver trains a parametric conditional distribution
    :math:`q_\phi(\theta | x)`  to approximate the posterior distribution
    :math:`p(\theta | x)` of parameters given observations.

    References
    ----------
    [1] Fast :math:`\espilon`-free Inference of Simulation Models with
        Bayesian Conditional Density Estimation
        (Papamakarios et al., 2016), https://arxiv.org/abs/1605.06376
    [2] Automatic posterior transformation for likelihood-free inference
        (Greenberg et al., 2019), https://arxiv.org/abs/1905.07488
    """

    name = "NPE"
    # Parameters of the methods. We can use this to change the embedding
    # network, the density estimator or other hyperparameters
    parameters = {
        "n_epochs": [2**31 - 1],
    }

    # test config: reduce the number of epochs
    test_config = {
        "n_epochs": 3,
    }

    def set_objective(self, thetas, xs, task):
        """Get information from the objective on the task considered.

        Parameters
        ----------
        thetas : torch.Tensor[num_train, d_params]
            The sampled parameters for training the posterior estimator
        xs : torch.Tensor[num_train, d_obs]
            The observations associated with the parameters thetas
        task : object
            The task object containing the prior distribution
        """
        self.thetas, self.xs, self.task = thetas, xs, task

    def run(self, n_iter: int):
        """Initialize and train the NPE"""

        # XXX: we could add a way to select the density estimator architecture
        self.npe = npe = NPE(self.task.get_prior_dist())
        npe.append_simulations(self.thetas, self.xs)
        npe.train(max_num_epochs=self.n_epochs)

    def get_result(self):
        """Returns the result of the method to the objective.

        Returns
        -------
        sample_func: Callable[[Tuple[int], torch.Tensor], torch.Tensor]
            A callable to sample the posterior associated to one observation.
        """
        posterior = self.npe.build_posterior()
        return {
            'sample_func': posterior.sample,
        }
