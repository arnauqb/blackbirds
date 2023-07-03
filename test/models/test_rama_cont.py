import torch
import numpy as np

from blackbirds.models.rama_cont import RamaCont


class TestRamaCont:
    def test__init(self):
        rc = RamaCont(n_agents=1000, n_timesteps=100, s=0.1, sigmoid_k=5.0)
        assert rc.n_agents == 1000
        assert rc.s == 0.1
        assert rc.sigmoid_k == 5.0
        assert rc.n_timesteps == 100

        params = torch.tensor([0.0, 1.0, 0.5, 0.2])  # a, b, sigma, eta
        u0_dist = torch.distributions.LogNormal(params[0], params[1])
        x = rc.initialize(params)
        assert x.shape == (1, 3, 1000)
        assert torch.isclose(x[:, 0].mean(), u0_dist.mean, rtol=1e-2)

    def test__order(self):
        rc = RamaCont(n_agents=100, n_timesteps=100, s=0.1, sigmoid_k=500.0)
        epsilon_t = torch.tensor(5.0)
        nu_i = torch.distributions.Uniform(0, 10).sample((rc.n_agents,))
        order_soft = rc.compute_order_soft(epsilon_t, nu_i)
        assert order_soft.shape == (100,)
        expected = ((epsilon_t > nu_i).int() - (epsilon_t < -nu_i).int()).float()
        not_match = len(torch.where(order_soft != expected)[0])
        assert not_match < 20
        order_hard = rc.compute_order_hard(epsilon_t, nu_i)
        assert order_hard.shape == (100,)
        assert torch.allclose(order_hard, expected)

        epsilon_t = torch.tensor(5.0, requires_grad=True)
        order = rc.compute_order(epsilon_t, nu_i)
        assert order.shape == (100,)
        assert torch.allclose(order, expected)
        assert order.requires_grad

    def test__rt(self):
        rc = RamaCont(n_agents=5, n_timesteps=100, s=0.1, sigmoid_k=5)
        order = torch.tensor([1.0, 2, 3, 4, 5])
        rets = rc.compute_returns(order=order, eta=0.5)
        assert torch.isclose(rets, torch.tensor(1 / (5 * 0.5) * 15))

    def test__update_nu_t(self):
        rc = RamaCont(n_agents=1000, n_timesteps=100, s=0.1, sigmoid_k=5)
        nu_t = torch.arange(0, 1000)
        s = torch.tensor(0.1)
        returns = torch.tensor(2.0)
        equal_to_rt = 0
        equal_to_nu_t = 0
        N = 100
        for _ in range(N):
            new_nu_t = rc.compute_new_nu_t(nu_t, s, returns)
            equal_to_rt += (new_nu_t == returns).float().sum() / rc.n_agents
            equal_to_nu_t += (new_nu_t == nu_t).float().sum() / rc.n_agents
        equal_to_rt /= N
        equal_to_nu_t /= N
        assert torch.isclose(equal_to_rt, s, atol=0.05)
        assert torch.isclose(equal_to_nu_t, 1 - s, atol=0.05)

    def test__runs(self):
        rc = RamaCont(n_agents=1000, n_timesteps=100, s=0.1, sigmoid_k=5.0)
        params = torch.tensor([0.1, 1.0, 0.5, 0.2])  # a, b, sigma, eta
        ts = rc.run_and_observe(params)
        assert ts[0].shape == (101,)
