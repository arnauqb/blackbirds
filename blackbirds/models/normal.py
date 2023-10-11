import torch

from blackbirds.models.model import Model


class Normal(Model):
    def __init__(
        self,
        n_timesteps: int,
    ):
        super().__init__()
        self.n_timesteps = n_timesteps

    def initialize(self, params):
        return torch.zeros(1).reshape(1, 1)

    def trim_time_series(self, x):
        return x[-1:]

    def step(self, params, x):
        mu, sigma = params
        assert sigma > 0, "Argument sigma must be a float greater than 0."
        return mu + sigma * torch.randn((1, 1))

    def observe(self, x):
        return [x]
