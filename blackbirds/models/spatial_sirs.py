import numpy as np
from numba import njit
import torch

from blackbirds.models.model import Model
from blackbirds.utils import soft_maximum, soft_minimum

@njit
def _count_infected_neighbours(x, i, j):

    N = x.shape[0]
    total = 0
    up = x[i - 1, j]
    if up == 1:
        total += 1
    down = x[(i + 1) % N, j]
    if down == 1:
        total += 1
    left = x[i, j - 1]
    if left == 1:
        total += 1
    right = x[i, j + 1]
    if right == 1:
        total += 1
    return total

@njit
def _update_cell(alpha, beta, gamma, x, i, j):

    cell_state = x[i, j]
    if cell_state == 0:
        n_infected_neighbours = _count_infected_neighbours(x, i, j)
        prob_infected = 1. - (1. - alpha)**n_infected_neighbours
        #prob_infected = alpha * n_infected_neighbours / 4.
        #prob_infected = 1. - np.exp(-n_infected_neighbours / 4. * alpha)
        #if n_infected_neighbours > 0:
        #    print(n_infected_neighbours, prob_infected)
        if np.random.random() < prob_infected:
            return 1
        return 0
    elif cell_state == 1:
        if np.random.random() < beta:
        #if np.random.random() < 1. - np.exp(-beta):
            return 2
        return 1
    elif cell_state == 2:
        if np.random.random() < gamma:
        #if np.random.random() < 1. - np.exp(-gamma):
            return 0
        return 2

@njit
def _update(alpha, beta, gamma, new_array, current_state):

    N = current_state.shape[0]
    for i in range(N):
        for j in range(N):
            #print(i, j)
            new_array[i, j] = _update_cell(alpha, beta, gamma, current_state, i, j)
    return new_array

class SIRS(Model):
    r"""Non-differentiable implementation of a spatial SIRS model on regular grid and periodic boundaries.

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
        # Grid size in each dimension
        self._N = N

    def initialize(self, params):
        state = torch.zeros((1, self._N, self._N))

        # Initialise uniformly at random
        idx = torch.randperm(self._N**2)[:int(self._i0 * self._N**2)]
        x_idx = idx // self._N
        y_idx = idx % self._N
        state[0, x_idx, y_idx] = 1

        #width = self._i0 * self._N
        #half_width = int(width / 2)
        #N_over_2 = int(self._N / 2)
        #state[0, 
        #      N_over_2 - half_width:N_over_2 + half_width, 
        #      N_over_2 - half_width:N_over_2 + half_width] = 1 
        
        self._t = 0
        return state

    def trim_time_series(self, x):
        return x[-1:]

    def step(self, params, x):
        r"""
        Runs the model forward for one time-step. Parameters follow the order: log_alpha, log_beta, log_gamma

        **Arguments:**

        - `params`: A list of parameters. Parameters follow the order: log_alpha, log_beta, log_gamma
        - `x`: The current state of the model.
        """
        alpha, beta, gamma = 1./torch.exp(params)
        assert (alpha > 0) & (alpha < 1)
        assert (beta > 0) & (beta < 1)
        assert (gamma > 0) & (gamma < 1)

        #print("==== t = {0} ====".format(self._t))
        self._t += 1
        #print(x)
        x_t = x.numpy().copy()[-1]
        # Because of this, the gradients aren't going to be right if we differentiate through the simulator
        x_t = _update(alpha.numpy(), beta.numpy(), gamma.numpy(), x_t, x.numpy()[-1])
        x_t = torch.from_numpy(x_t).unsqueeze(0)
        #print(x_t, x_t.shape)
        #print()
        return x_t

    def observe(self, x):
        return [x.unsqueeze(-1)]
