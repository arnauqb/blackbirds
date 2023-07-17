import torch

from blackbirds.models.model import Model
from blackbirds.utils import soft_maximum, soft_minimum


class SIRS(Model):
    r"""Differentiable implementation of the mean-field SIRS model.

    **Arguments:**

    - `n_timesteps`: Number of timesteps to simulate. Default: 100.
    - `device`: Device to run on. Default 'cpu'
    - `i0`: Initial proportion of population to be infected. \in (0,1). Everything else is 0.

    WARNING: GRADIENT WON'T BE CORRECT IN THIS MODEL SINCE I DIDN'T MAKE IT CORRECT
    """

    def __init__(self, n_timesteps=100, device="cpu", i0=0.1, N=100):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.device = device
        self._i0 = i0
        self._state = torch.zeros((3,1))
        self._N = N
        self._state[1] = int(self._i0 * self._N)

    def initialize(self, params):
        return torch.zeros((3, 1))

    def trim_time_series(self, x):
        return x[-1:]

    def step(self, params, x):
        r"""
        Runs the model forward for one time-step. Parameters follow the order: log_alpha, log_beta, log_gamma

        **Arguments:**

        - `params`: A list of parameters. Parameters follow the order: log_alpha, log_beta, log_gamma
        - `x`: The current state of the model.
        """
        alpha, beta, gamma = torch.exp(params)

        S, I, R = self._state
        x_t = torch.zeros((3, 1))
        # Because of this, the gradients aren't going to be right if we differentiate through the simulator
        x_t[0] = torch.distributions.binomial.Binomial(R, probs=1. - torch.exp([-gamma])).sample((1,))
        x_t[1] = torch.distributions.binomial.Binomial(S, probs=1. - torch.exp([-alpha * I / self._N])).sample((1,))
        x_t[2] = torch.distributions.binomial.Binomial(I, probs=1. - torch.exp([-beta])).sample((1,))
        self._state[0] = S + x_t[0] - x_t[1]
        self._state[1] = I + x_t[1] - x_t[2]
        self._state[2] = R + x_t[2] - x_t[0]
        return x_t.reshape(1, -1)

    def observe(self, x):
        return [x]

    def reconstruct_series(self, x):
        """
        Assumes x is a torch.Tensor of shape (T, 3) denoting the number of transitions between
        each of three compartments at each time step
        """
        init_state = torch.zeros((3,1))
        init_state[1] = int(self._i0 * self._N)
        series = [init_state]
        for t in range(len(x)):
            current_state = series[-1].copy()
            current_state[0] += x[t][0] - x[t][1]
            current_state[1] += x[t][1] - x[t][2]
            current_state[2] += x[t][2] - x[t][0]
            series.append(current_state)
        return series

    def negative_log_likelihood(self, params, y):
     
        alpha, beta, gamma = torch.exp(params)
        series = self.reconstruct_series(y)
        binomial = torch.distributions.binomial.Binomial
        ll = 0.
        for t in range(len(y)):
            S, I, R = series[t]
            ll_rs = binomial(R, probs=1. - torch.exp([-gamma])).log_prob(y[t][0])
            ll_si = binomial(S, probs=1. - torch.exp([-alpha * I / self._N])).log_prob(y[t][1])
            ll_ir = binomial(I, probs=1. - torch.exp([-beta])).log_prob(y[t][2])
            ll += ll_rs + ll_si + ll_ir
        return -ll
