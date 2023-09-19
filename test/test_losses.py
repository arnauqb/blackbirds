import numpy as np
import torch

from blackbirds.losses import (
    SingleOutput_SimulateAndMSELoss,
    SingleOutput_SimulateAndMMD,
    UnivariateMMDLoss,
)
from blackbirds.models.normal import Normal
from blackbirds.simulate import simulate_and_observe_model


class TestMSELoss:
    def test_normal_same(self):
        T = 5000
        normal = Normal(n_timesteps=T)
        mu, sigma = 1.0, 2.0
        params = torch.Tensor([mu, sigma])
        y = simulate_and_observe_model(normal, params)[0]
        mse_loss = SingleOutput_SimulateAndMSELoss(normal)
        loss_value = mse_loss(params, y)
        # MMD between distributions of hould be close to...
        value = 2 * sigma**2
        assert np.isclose(loss_value, value, atol=1e-2, rtol=5e-2)

    def test_normal_different(self):
        T = 5000
        normal = Normal(n_timesteps=T)
        mu_y, sigma_y = 1.0, 2.0
        params = torch.Tensor([mu_y, sigma_y])
        y = simulate_and_observe_model(normal, params)[0]
        mse_loss = SingleOutput_SimulateAndMSELoss(normal)
        mu_x, sigma_x = 0.0, 1.0
        params = torch.Tensor([mu_x, sigma_x])
        loss_value = mse_loss(params, y)
        # MMD between distributions of hould be close to...
        value = sigma_x**2 + sigma_y**2 + (mu_x - mu_y) ** 2
        assert np.isclose(loss_value, value, atol=1e-2, rtol=5e-2)


class TestUnivariateMMDLoss:
    def test_normal_same(self):
        T = 500
        normal = Normal(n_timesteps=T)
        mu, sigma = 1.0, 2.0
        params = torch.Tensor([mu, sigma])
        y = simulate_and_observe_model(normal, params)[0]
        mmd_loss = SingleOutput_SimulateAndMMD(y, normal)
        loss_value = mmd_loss(params, y)
        # MMD between distributions of hould be close to 0
        assert np.isclose(loss_value, 0.0, atol=1e-3)

    def test_normal_different(self):
        T = 500
        normal = Normal(n_timesteps=T)
        mu, sigma = 1.0, 2.0
        params = torch.Tensor([mu, sigma])
        y = simulate_and_observe_model(normal, params)[0]
        mmd_loss = SingleOutput_SimulateAndMMD(y, normal)
        mu, sigma = 0.0, 1.0
        params = torch.Tensor([mu, sigma])
        loss_value = mmd_loss(params, y)
        # MMD between distributions of hould be close to 0
        assert not np.isclose(loss_value, 0.0)
        assert loss_value > 0.0