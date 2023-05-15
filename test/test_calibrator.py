import torch
import numpy as np

from birds.models.random_walk import RandomWalk
from birds.calibrator import Calibrator


class TrainableGaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(0.5 * torch.ones(1))
        self.sigma = torch.nn.Parameter(0.1 * torch.ones(1))

    def log_prob(self, x):
        sigma = torch.clip(self.sigma, min=1e-3)
        return torch.distributions.Normal(self.mu, sigma).log_prob(x)

    def rsample(self, x):
        sigma = torch.clip(self.sigma, min=1e-3)
        return torch.distributions.Normal(self.mu, sigma).rsample(x)

    def sample(self, x):
        sigma = torch.clip(self.sigma, min=1e-3)
        return torch.distributions.Normal(self.mu, sigma).sample(x)


class TestCalibrator:
    def test_random_walk(self):
        """
        Tests inference in a random walk model.
        """
        rw = RandomWalk(100)
        true_ps = [0.25, 0.5, 0.75]
        prior = torch.distributions.Normal(0.0, 1.0)
        for diff_mode in ("reverse", "forward"):
            for true_p in true_ps:
                data = rw(torch.tensor([true_p]))
                posterior_estimator = TrainableGaussian()
                optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=5e-2)
                calib = Calibrator(
                    model=rw,
                    posterior_estimator=posterior_estimator,
                    prior=prior,
                    data=data,
                    optimizer=optimizer,
                    diff_mode=diff_mode,
                )
                _, best_model_state_dict = calib.run(1000)
                posterior_estimator.load_state_dict(best_model_state_dict)
                assert np.isclose(posterior_estimator.mu.item(), true_p, rtol=0.25)
                assert posterior_estimator.sigma.item() < 1e-2
