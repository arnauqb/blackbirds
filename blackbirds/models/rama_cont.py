import torch

from blackbirds.models.model import Model


class RamaCont(Model):
    def __init__(self, n_agents, n_timesteps, s, sigmoid_k):
        """
        Implementation of the Rama Cont model from Rama Cont (2005).

        **Arguments**

        - `n_agents`: Number of agents
        - `n_timesteps`: Number of timesteps
        - `s`: Probability of updating the threshold $\nu_i$.
        - `sigmoid_k`: Steepness of the sigmoid function.
        """
        super().__init__()
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.s = s
        self.sigmoid_k = sigmoid_k

    def initialize(self, params):
        nu_0 = torch.distributions.LogNormal(params[0], params[1]).rsample(
            (self.n_agents,)
        )
        epsilon_t = torch.zeros(self.n_agents)
        order = self.compute_order(epsilon_t, nu_0)
        eta = params[3]
        returns = self.compute_returns(order, eta) * torch.ones(self.n_agents)
        x = torch.vstack((nu_0, epsilon_t, returns))
        return x.reshape(1, 3, self.n_agents)

    def step(self, params, x):
        # draw epsilon_t from normal distribution
        sigma = params[2]
        epsilon_t = torch.distributions.Normal(0, sigma).rsample((self.n_agents,))
        # compute order
        nu_t = x[-1, 0, :]
        order = self.compute_order(epsilon_t, nu_t)
        # compute returns
        eta = params[3]
        returns = self.compute_returns(order, eta)
        # update nu_t
        new_nu_t = self.compute_new_nu_t(nu_t, self.s, returns)
        x = torch.vstack(
            (
                new_nu_t,
                epsilon_t * torch.ones(self.n_agents),
                returns * torch.ones(self.n_agents),
            )
        )
        return x.reshape(1, 3, self.n_agents)

    def observe(self, x):
        return [x[:, 2, 0]]

    def compute_order_soft(self, epsilon_t, nu_t):
        return torch.sigmoid(self.sigmoid_k * (epsilon_t - nu_t)) - torch.sigmoid(
            self.sigmoid_k * (-nu_t - epsilon_t)
        )

    def compute_order_hard(self, epsilon_t, nu_t):
        return (epsilon_t > nu_t).float() - (epsilon_t < -nu_t).float()

    def compute_order(self, epsilon_t, nu_t):
        soft = self.compute_order_soft(epsilon_t, nu_t)
        return self.compute_order_hard(epsilon_t, nu_t) + soft - soft.detach()

    def compute_returns(self, order, eta):
        return 1.0 / (self.n_agents * eta) * order.sum()

    def compute_new_nu_t(self, nu_t, s, returns):
        probs = s * torch.ones(self.n_agents)
        probs = torch.vstack((probs, 1.0 - probs)).transpose(0, 1)
        q = torch.nn.functional.gumbel_softmax(probs.log(), tau=0.1, hard=True)[:, 0]
        return torch.abs(returns) * q + (1 - q) * nu_t

    def trim_time_series(self, x):
        return x
