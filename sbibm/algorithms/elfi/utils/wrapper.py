import numpy as np
import torch


class Simulator:
    def __init__(self, simulator):
        self.simulator = simulator

    def __call__(self, *args, batch_size=1, random_state=None):
        return (
            self.simulator(torch.from_numpy(np.stack(args).astype(np.float32).T))
            .numpy()
            .reshape(batch_size, -1)
        )
