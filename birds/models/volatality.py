import torch
import torch.nn as nn
from birds.models.model import Model

print("imports done")

'''
Constraints of the simulation:
1. Single stock
2. No social learning between agents: no fundamentalist or chartist distinction if agents
3. All agents receive same information, just differ in how they process it.
4. Agent state is the news acceptance threshold
5. Simulation parameters: i) D -> std of noise in news arrival. parameterizes \eps, ii) q -> avg updating frequency iii) \lamda -> market depth, which govern price setting
'''

class ClusteredVolatality(Model):
    def __init__(self, n_timesteps=2, n_agents=1000, device='cpu') -> None:
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_agents = n_agents
        self.device = device

        self.D = 0.01
        self._eps = torch.distributions.normal.Normal(
            torch.tensor([0.0], device=device), torch.tensor([self.D**2], device=device)
        )
        self._u = torch.distributions.uniform.Uniform(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))


    def initialize(self, params):
        threshold = torch.zeros((1, self.n_agents))

        return threshold

    def step(self, params, x):
        r"""
        1. Each agent receives common external news signal \epsilon
        2. Each agent compares \epsilon to their threshold \theta_i(t)
        3. Based on threshold, agent makes a purchase order (\phi_i).
        4. Market price is impacted based on excess order (Z = sum(\phi_i)). This produces a new price parameterized by \lamda
        5. With probability q, agent updates their threshold \theta_i{t}
        """

        q_val = params[0]
        lam = params[1]

        epsilon_signal = self._eps.rsample() 
        order_phi = 1*(epsilon_signal > x[-1]) - 1*(epsilon_signal < -x[-1])
        excess_demand = order_phi.sum()

        step_return = torch.atan(excess_demand/lam)

        u_sample = self._u.sample((self.n_agents,))
        update_mask = (q_val > u_sample).long().view(-1)

        new_x = update_mask*torch.abs(step_return) + (1 - update_mask)*x[-1]

        return new_x.view(1, -1)

