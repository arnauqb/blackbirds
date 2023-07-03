import torch


class TrainableGaussian(torch.nn.Module):
    def __init__(self, mu=[0.0], sigma=1.0, device="cpu"):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(mu, device=device))
        self.sigma = sigma * torch.eye(len(mu), device=device)
        self.sigma = torch.nn.Parameter(self.sigma)

    def clamp_sigma(self):
        sigma = self.sigma.clone()
        mask = torch.eye(len(self.mu), device=self.sigma.device).bool()
        sigma[mask] = torch.clamp(self.sigma[mask], min=1e-3)
        return sigma

    def log_prob(self, x):
        sigma = self.clamp_sigma()
        return torch.distributions.MultivariateNormal(self.mu, sigma).log_prob(x)

    def sample(self, n):
        sigma = self.clamp_sigma()
        dist = torch.distributions.MultivariateNormal(self.mu, sigma)
        sample = dist.rsample((n,))
        return sample, self.log_prob(sample.detach())

    def __call__(self, x=None):
        return self
