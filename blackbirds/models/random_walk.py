import torch

from blackbirds.models.model import Model


class RandomWalk(Model):
    def __init__(self, n_timesteps, tau_softmax=0.1):
        r"""Implements a differentiable random walk.

        $$
            X_t = \sum_{i=1}^t (2\eta - 1),
        $$

        where

        $$
        \eta \sim \text{Bernoulli}(p).
        $$

        **Arguments**:

        - `n_timesteps` (int): Number of timesteps to simulate.
        - `tau_softmax` (float): Temperature parameter for the Gumbel-Softmax
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.tau_softmax = tau_softmax

    def initialize(self, params):
        return torch.zeros(1).reshape(1, 1)

    def trim_time_series(self, x):
        return x[-1:]

    def step(self, params, x):
        """Simulates a random walk step using the Gumbel-Softmax trick.

        **Arguments:**

        - `params`: a tensor of shape (1,) containing the logit probability of moving forward at each timestep.
        - `x`: a tensor of shape (n,) containing the time-series of positions.

        !!! danger
            probability is given in logit, so the input is transformed using the sigmoid function.
        """
        p = torch.sigmoid(params)
        logits = torch.vstack((p, 1 - p)).log()
        step = torch.nn.functional.gumbel_softmax(
            logits, dim=0, tau=self.tau_softmax, hard=True
        )
        return (x[-1] + step[0] - step[1]).reshape(1, 1)

    def observe(self, x):
        return [x]
