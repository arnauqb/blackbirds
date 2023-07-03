import torch

from blackbirds.models.model import Model


class Autoregressive(Model):
    def __init__(self, n_timesteps: int, sigma_eps: float = 1.0):
        r"""
        Implements an autoregressive model of order `p`.

        $$
        X_t = \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t
        $$

        **Arguments:**

        - `n_timesteps`: The number of timesteps to predict into the future.
        - `sigma_eps`: The standard deviation of the gaussian noise.
        """
        self.n_timesteps = n_timesteps
        self.sigma_eps = torch.tensor(sigma_eps)

    def initialize(self, params):
        """
        Initializes the model with the given parameters.

        The fist time-step is just some random gaussian noise.
        """
        # return random noise
        return torch.normal(0.0, self.sigma_eps).reshape(1, 1)

    def step(self, params, x):
        """
        Implements a single step of the autoregressive model.

        **Arguments:**

        - `params`: The parameters of the model.
        - `x`: The current state of the model.
        """
        n_past_timesteps = min(x.shape[0], params.shape[0])
        return torch.sum(
            params[:n_past_timesteps] * x[-n_past_timesteps:]
        ) + torch.normal(0.0, self.sigma_eps).reshape(1, 1)

    def observe(self, x):
        return [x]

    def trim_time_series(self, x):
        return x
