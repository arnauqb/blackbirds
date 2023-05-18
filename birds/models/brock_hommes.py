import torch
import torch.distributions as distributions


class BrockHommes(torch.nn.Module):
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

    def forward(self, params):
        """
        Runs the model forward. Parameters follow the order: log_beta, g1, g2, g3, g4, b1, b2, b3, b4, log_sigma, log_r

        !!! danger
        beta, sigma, and r are given in log.
        """
        beta = torch.exp(params[0])
        g = params[1:5]
        b = params[5:9]
        sigma = torch.exp(params[-2])
        r = torch.exp(params[-1])
        R = 1.0 + r

        epsilons = self._eps.rsample((self.n_timesteps,))
        x = torch.zeros(3)
        for t in range(self.n_timesteps):
            exponent = (
                beta * (x[t - 1] - R * x[t - 2]) * (g * x[t - 3] + b - R * x[t - 2])
            )
            norm_exponentiated = torch.nn.Softmax(dim=-1)(exponent)
            mean = (norm_exponentiated * (g * x[t - 1] + b)).sum()
            x_t = (mean + epsilons[t] * sigma) / R
            x = torch.hstack((x, x_t))
        return [x[3:]]
