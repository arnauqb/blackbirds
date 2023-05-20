import torch

from birds.models.model import Model


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

        Arguments:
            n_timesteps (int): Number of timesteps to simulate.
            tau_softmax (float): Temperature parameter for the Gumbel-Softmax
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.tau_softmax = tau_softmax

    def initialize(self, params):
        return torch.zeros(1)

    def step(self, params, x):
        """Simulates a random walk step using the Gumbel-Softmax trick.

        **Arguments:**

        - params: a tensor of shape (1,) containing the probability of moving forward at each timestep.
        - x: a tensor of shape (n,) containing the time-series of positions.
        """
        p = torch.clip(params, min=0.0, max=1.0)
        logits = torch.vstack((p, 1 - p)).log()
        step = torch.nn.functional.gumbel_softmax(
            logits, dim=0, tau=self.tau_softmax, hard=True
        )
        return x[-1] + step[0] - step[1]

    def run(self, params):
        p = torch.clip(params, min=0.0, max=1.0)
        p = p * torch.ones(self.n_timesteps)
        logits = torch.vstack((p, 1 - p)).log()
        steps = torch.nn.functional.gumbel_softmax(
            logits, dim=0, tau=self.tau_softmax, hard=True
        )
        forward_steps = steps[0,:]
        backward_steps = steps[1,:]
        x = forward_steps - backward_steps
        x = torch.hstack((torch.zeros([1]), x))
        return x.reshape(-1, 1)

    def observe(self, x):
        return [x.cumsum(dim=0)]
