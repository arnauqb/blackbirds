import torch
import torch.distributions as distributions

from birds.models.model import Model


class BrockHommes(Model):
    r"""Differentiable implementation of the Brock and Hommes (1998) model. See equations (39) and (40)
    of https://arxiv.org/pdf/2202.00625.pdf for reference.

    **Arguments:**

    - `n_timesteps`: Number of timesteps to simulate. Default: 100.
    """

    def __init__(self, n_timesteps=100, device="cpu"):
        super().__init__()
        self.n_timesteps = n_timesteps
        self._eps = distributions.normal.Normal(
            torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
        )
        self.device = device

    def initialize(self, params):
        return torch.zeros(3)

    def step(self, params, x):
        r"""
        Runs the model forward for one time-step. Parameters follow the order: log_beta, g1, g2, g3, g4, b1, b2, b3, b4, log_sigma, log_r

        **Arguments:**

        - `params`: A list of parameters. Parameters follow the order: log_beta, g1, g2, g3, g4, b1, b2, b3, b4, log_sigma, log_r
        - `x`: The current state of the model.

        !!! danger
        beta, sigma, and r are given in log.
        """
        beta = torch.exp(params[0])
        g = params[1:5]
        b = params[5:9]
        sigma = torch.exp(params[-2])
        r = torch.exp(params[-1])
        R = 1.0 + r
        
        epsilon = self._eps.rsample()
        exponent = (
            beta * (x[-1] - R * x[-2]) * (g * x[-3] + b - R * x[-2])
        )
        norm_exponentiated = torch.nn.Softmax(dim=-1)(exponent)
        mean = (norm_exponentiated * (g * x[-1] + b)).sum()
        x_t = (mean + epsilon * sigma) / R
        return x_t

    def observe(self, x):
        return [x]
