from abc import ABC, abstractmethod

import torch


class Model(ABC):
    @abstractmethod
    def initialize(self, params):
        pass

    @abstractmethod
    def step(self, params, x):
        pass

    def __call__(self, params, x):
        return self.step(params, x)

    @abstractmethod
    def observe(self, x):
        pass

    @abstractmethod
    def trim_time_series(self, time_series):
        pass

    def run(self, params):
        x = self.initialize(params)
        for _ in range(self.n_timesteps):
            x_t = self.step(params, x)
            x = torch.vstack((x, x_t))
        return x
