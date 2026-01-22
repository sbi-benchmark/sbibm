from benchopt import BaseDataset

import sbibm


class Dataset(BaseDataset):
    """Linear Gaussian dataset.

    Used for the tests of all solvers.
    """

    name = "gaussian_linear"
    parameters = {
        'num_observation': [1],
        'num_samples': [10_000],
    }

    test_parameters = {
        'num_samples': [100]
    }

    def get_data(self):
        r"""Generate data.

        Returns the input of the `Objective.set_data` method.
        """
        task = sbibm.get_task("gaussian_linear")

        # Sample a training set and get the reference observation
        thetas = task.get_prior()(self.num_samples)
        xs = task.get_simulator()(thetas)
        obs_ref = task.get_observation(self.num_observation)
        theta_ref = task.get_reference_posterior_samples(self.num_observation)
        return {
            'task': task,
            'thetas': thetas,
            'xs': xs,
            'theta_ref': theta_ref,
            'obs_ref': obs_ref,
        }
