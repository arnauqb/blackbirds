import torch
import torch.autograd as autograd
import numpy as np
from blackbirds.models.normal import Normal


class TestRandomWalk:
    def test__result(self):
        torch.manual_seed(0)
        n_timesteps = 5000
        normal = Normal(n_timesteps)
        assert normal.n_timesteps == n_timesteps
        mu, sigma = 1., 3.
        params = torch.Tensor([mu, sigma])
        x = normal.run(params)
        assert x.shape == (n_timesteps + 1, 1)
        trajectory = normal.observe(x)[0]
        mean = trajectory.mean()
        std = trajectory.std()
        # Sample mean should approximately match mu
        assert np.isclose(
            mean, mu, atol=1e-1
        )
        # Sample std should approximately match sigma
        assert np.isclose(
            std, sigma, rtol=1e-1
        )
        # Should also have approximately 0. skewness
        sample_skewness = torch.pow((trajectory - mean)/std, 3.).mean()
        assert np.isclose(
            sample_skewness, 0., atol=1e-1
        )

    def test__gradient(self):
        mu, sigma = 0.4, 1.5
        params = torch.Tensor([mu, sigma])
        params.requires_grad = True
        n_timesteps = 100
        normal = Normal(n_timesteps)
        assert normal.n_timesteps == n_timesteps
        x = normal.observe(normal.run(params))[0]
        assert x.shape == (n_timesteps + 1, 1)
        x.sum().backward()
        assert params.grad is not None
        assert np.isclose(params.grad[0], n_timesteps)
        grad_sigmas = []

        def mean_x(params):
            x = normal.observe(normal.run(params))[0]
            return x.mean()

        for t in range(500): 
            params = torch.Tensor([mu, sigma])
            params.requires_grad = True
            sigma_grad = autograd.grad(mean_x(params), params)[-1]
            grad_sigmas.append(sigma_grad[-1])

        assert np.isclose(torch.stack(grad_sigmas).mean(), 0., atol=1e-2)
        expected_std = 1. / np.sqrt(n_timesteps)
        assert np.isclose(torch.stack(grad_sigmas).std().item(), expected_std, atol=1e-2)
