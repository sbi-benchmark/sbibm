from typing import Any, Callable, Optional

import torch

from sbibm.tasks.task import Task
from sbibm.utils.exceptions import SimulationBudgetExceeded


class Simulator:
    def __init__(
        self,
        task: Task,
        simulator: Callable,
        max_calls: Optional[int] = None,
    ):
        """Simulator

        Each task defines a simulator and passes it into this class, which wraps it.
        When a simulator is called with parameters, the `__call__` method of this
        class is invoked.

        `__call__` simply forwards the parameters to the simulator function, while
        checking parameter dimensions and increasing an internal counter. The internal
        counter ensures that a simulator can only be called a certain maximum number
        of times, if a limit is set through `max_calls`.

        Args:
            task: Task instance, used to read out properties such as dimensionality of
                parameters and data, as well as the name
            simulator: The simulator defined by the task
            max_calls: If set, limits calls to simulator before an error is raised
        """
        self.simulator = simulator
        self.max_calls = max_calls
        self.num_simulations = 0

        self.name = task.name
        self.dim_data = task.dim_data
        self.dim_parameters = task.dim_parameters
        self.flatten_data = task.flatten_data
        self.unflatten_data = task.unflatten_data

    def __call__(self, parameters: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)

        assert parameters.ndim == 2
        assert parameters.shape[1] == self.dim_parameters

        requested_simulations = parameters.shape[0]

        if (
            self.max_calls is not None
            and self.num_simulations + requested_simulations > self.max_calls
        ):
            raise SimulationBudgetExceeded

        data = self.simulator(parameters, **kwargs)

        self.num_simulations += requested_simulations

        return self.flatten_data(data)
