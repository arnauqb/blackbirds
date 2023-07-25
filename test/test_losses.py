import numpy as np
import torch

from blackbirds.losses import SingleOutput_SimulateAndMSELoss, SingleOutput_SimulateAndMMD, MMDLoss, UnivariateMMDLoss
from blackbirds.models.normal import Normal
from blackbirds.models.random_walk import RandomWalk
from blackbirds.simulate import simulate_and_observe_model

class TestMSELoss:

    def test_normal_same(self):

        T = 5000
        normal = Normal(n_timesteps=T)
        mu, sigma = 1., 2.
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
        mu_y, sigma_y = 1., 2.
        params = torch.Tensor([mu_y, sigma_y])
        y = simulate_and_observe_model(normal, params)[0]
        mse_loss = SingleOutput_SimulateAndMSELoss(normal)
        mu_x, sigma_x = 0., 1.
        params = torch.Tensor([mu_x, sigma_x])
        loss_value = mse_loss(params, y)
        # MMD between distributions of hould be close to...
        value = sigma_x**2 + sigma_y**2 + (mu_x - mu_y)**2
        assert np.isclose(loss_value, value, atol=1e-2, rtol=5e-2)


class TestMMDLoss:

    def test_normal_same(self):

        T = 500
        normal = Normal(n_timesteps=T)
        mu, sigma = 1., 2.
        params = torch.Tensor([mu, sigma])
        y = simulate_and_observe_model(normal, params)[0]
        mmd_loss = SingleOutput_SimulateAndMMD(y, normal)
        loss_value = mmd_loss(params, y)
        # MMD between distributions of hould be close to 0
        assert np.isclose(loss_value, 0., atol=1e-3)

    def test_normal_different(self):

        T = 500
        normal = Normal(n_timesteps=T)
        mu, sigma = 1., 2.
        params = torch.Tensor([mu, sigma])
        y = simulate_and_observe_model(normal, params)[0]
        mmd_loss = SingleOutput_SimulateAndMMD(y, normal)
        mu, sigma = 0., 1.
        params = torch.Tensor([mu, sigma])
        loss_value = mmd_loss(params, y)
        # MMD between distributions of hould be close to 0
        assert not np.isclose(loss_value, 0.)
        assert loss_value > 0.


class TestMMDLoss:
    def test__compute_pairwise_distances(self):
        x = torch.randn(100, 2)
        y = torch.randn(50, 2)
        loss = MMDLoss(y)
        loss_calc = loss._pairwise_distance(x, y)
        x_cdist = x.view(1, 100, 2)
        y_cdist = y.view(1, 50, 2)
        torch_calc = torch.cdist(x_cdist, y_cdist)[0,:,:]
        assert torch.allclose(torch_calc, loss_calc)

    def test__mmd_loss(self):
        X = torch.randn(100, 1)
        Y = torch.randn(50, 1)
        loss = MMDLoss(Y)
        loss_1d = UnivariateMMDLoss(Y.flatten())
        assert torch.isclose(loss(X), loss_1d(X.flatten()))


