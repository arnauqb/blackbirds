from pytest import fixture
import numpy as np
import random
import torch


@fixture(autouse=True)
def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


class TrainableGaussian(torch.nn.Module):
    def __init__(self, mu=0.0, sigma=1.0):
        super().__init__()
        self.mu = torch.nn.Parameter(mu * torch.ones(1))
        self.sigma = torch.nn.Parameter(sigma * torch.ones(1))

    def log_prob(self, x):
        sigma = torch.clip(self.sigma, min=1e-3)
        return torch.distributions.Normal(self.mu, sigma).log_prob(x)

    def sample(self, x):
        sigma = torch.clip(self.sigma, min=1e-3)
        dist = torch.distributions.Normal(self.mu, sigma)
        sample = dist.rsample((x,))
        return sample, dist.log_prob(sample)


@fixture(name="TrainableGaussian")
def make_trainable_gaussian():
    return TrainableGaussian
