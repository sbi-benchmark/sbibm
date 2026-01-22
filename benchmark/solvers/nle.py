from benchopt import BaseSolver

from sbi.inference import NLE


class Solver(BaseSolver):
    """Neural likelihood estimation (NLE)"""

    name = "NLE"
    # Parameters of the methods. We can use this to change the embedding
    # network, the likelihood estimator or other hyperparameters
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
        self.nle = nle = NLE(self.task.get_prior_dist())
        nle.append_simulations(self.thetas, self.xs)

        nle.train(max_num_epochs=self.n_epochs)

    def get_result(self):
        """Returns the result of the method to the objective.

        Returns
        -------
        sample_func: Callable[[Tuple[int], torch.Tensor], torch.Tensor]
            A callable to sample the posterior associated to one observation.
        """
        posterior = self.nle.build_posterior()
        return {
            'sample_func': posterior.sample,
        }
