import torch

from birds.posterior_estimators import TrainableGaussian


class TestTrainableGaussian:
    def test__trainable_gaussian(self):
        tg = TrainableGaussian()
        assert tg.log_prob(torch.tensor([0.0])) == -0.9189385332046509
        assert tg.log_prob(torch.tensor([1.0])) == -1.4189385175704956
        samples = tg.sample(10000)[0]
        assert torch.isclose(
            samples.flatten().mean(), torch.tensor(0.0), rtol=1e-3, atol=1e-2
        )

        tg = TrainableGaussian(mu=[1.0, 2.0, 3.0], sigma=2.0)
        dist = torch.distributions.MultivariateNormal(
            torch.tensor([1.0, 2.0, 3.0]), 2.0 * torch.eye(3)
        )
        shouldbe = tg.log_prob(torch.tensor([1.0, 2.0, 3.0]))
        correct = dist.log_prob(torch.tensor([1.0, 2, 3]))
        assert torch.isclose(shouldbe, correct)
