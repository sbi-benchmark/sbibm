from typing import Callable
import torch
import numpy as np
from abcpy.continuousmodels import Continuous, InputConnector
from abcpy.distances import Euclidean, LogReg, PenLogReg
from abcpy.output import Journal
from abcpy.probabilisticmodels import ProbabilisticModel
from abcpy.statistics import Statistics


class ABCpyPrior(ProbabilisticModel, Continuous):
    def __init__(self, task, name='ABCpy_prior'):
        self.prior_forward = task.get_prior()
        self.dim_parameters = task.dim_parameters
        self.name = task.name if task.name is not None else name

        input_parameters = InputConnector.from_list([])
        super(ABCpyPrior, self).__init__(input_parameters, self.name + "_prior")

    def forward_simulate(self, abcpy_input_values, num_forward_simulations, rng=np.random.RandomState()):
        result = np.array(self.prior_forward(num_forward_simulations))
        return [np.array([x]).reshape(-1, ) for x in result]

    def get_output_dimension(self):
        return self.dim_parameters

    def _check_input(self, input_values):
        return True

    def _check_output(self, values):
        return True


class ABCpySimulator(ProbabilisticModel, Continuous):
    def __init__(self, parameters, task, max_calls, name='ABCpy_simulator'):
        self.simulator = task.get_simulator(max_calls=max_calls)
        self.output_dim = task.dim_data
        self.name = task.name if task.name is not None else name

        input_parameters = InputConnector.from_list(parameters)
        super(ABCpySimulator, self).__init__(input_parameters, self.name)

    def forward_simulate(self, abcpy_input_values, num_forward_simulations, rng=np.random.RandomState()):
        tensor_param = torch.tensor(abcpy_input_values)
        tensor_res = [self.simulator(tensor_param) for k in range(num_forward_simulations)]
        # print(tensor_res)
        # print([np.array(x).reshape(-1, ) for x in tensor_res])
        return [np.array(x).reshape(-1, ) for x in tensor_res]

    def get_output_dimension(self):
        return self.output_dim

    def _check_input(self, input_values):
        return True

    def _check_output(self, values):
        return True


def get_distance(distance: str, statistics: Statistics) -> Callable:
    """Return distance function for ABCpy."""

    if distance == "l2":
        distance_calc = Euclidean(statistics)

    elif distance == "log_reg":
        distance_calc = LogReg(statistics)

    elif distance == "pen_log_reg":
        distance_calc = PenLogReg(statistics)

    elif distance == "Wasserstein":
        raise NotImplementedError("Wasserstein distace not yet implemented as we are considering only one single "
                                  "simulation for parameter value")

    else:
        raise NotImplementedError(f"Distance '{distance}' not implemented.")

    return distance_calc


def journal_cleanup_rejABC(journal, percentile=None, threshold=None):
    """This function takes a Journal file (typically produced by an Rejection ABC run with very large epsilon value)
    and the keeps only the samples which achieve performance less than either some percentile of the achieved distances,
     or either some specified threshold. It is
    a very simple way to obtain a Rejection ABC which works on a percentile of the obtained distances. """

    if (threshold is None) == (percentile is None):
        raise RuntimeError("Exactly one of percentile or epsilon needs to be specified.")

    if percentile is not None:
        distance_cutoff = np.percentile(journal.distances[-1], percentile)
    else:
        distance_cutoff = threshold
    picked_simulations = journal.distances[-1] < distance_cutoff
    new_distances = journal.distances[-1][picked_simulations]
    if len(picked_simulations) == 0:
        raise RuntimeError("The specified value of threshold is too low, no simulations are selected.")

    new_journal = Journal(journal._type)
    new_journal.configuration["n_samples"] = journal.configuration["n_samples"]
    new_journal.configuration["n_samples_per_param"] = journal.configuration["n_samples_per_param"]
    new_journal.configuration["epsilon"] = journal.configuration["epsilon"]

    n_reduced_samples = np.sum(picked_simulations)

    new_accepted_parameters = []
    param_names = journal.get_parameters().keys()
    new_names_and_parameters = {name: [] for name in param_names}
    for i in range(len(picked_simulations)):
        if picked_simulations[i]:
            new_accepted_parameters.append(journal.get_accepted_parameters()[i])
            for name in param_names:
                new_names_and_parameters[name].append(journal.get_parameters()[name][i])

    new_journal.add_accepted_parameters(new_accepted_parameters)
    new_journal.add_weights(np.ones((n_reduced_samples, 1)))
    new_journal.add_ESS_estimate(np.ones((n_reduced_samples, 1)))
    new_journal.add_distances(new_distances)
    new_journal.add_user_parameters(new_names_and_parameters)
    new_journal.number_of_simulations.append(journal.number_of_simulations[-1])

    return new_journal
