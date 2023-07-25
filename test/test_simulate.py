import torch
import numpy as np

from blackbirds.simulate import (
    simulate_and_observe_model,
    compute_loss,
    generate_training_data,
)


class TestSimulate:
    def test__simulate_and_observe_model(self, mock_model):
        params = torch.tensor([2.0], requires_grad=True)
        observations = simulate_and_observe_model(
            model=mock_model, params=params, gradient_horizon=0
        )
        assert len(observations) == 1
        assert (observations[0] == torch.tensor([4.0, 4.0, 4.0])).all()
        y = torch.ones(3, 1)
        loss = torch.nn.MSELoss()(observations[0], y)
        loss.backward()
        assert params.grad is not None

    def test__compute_loss(self):
        loss_fn = torch.nn.MSELoss()
        observed_outputs = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6])]
        simulated_outputs = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6])]
        assert compute_loss(loss_fn, observed_outputs, simulated_outputs) == (0, 0)
        simulated_outputs = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 7])]
        assert torch.isclose(
            compute_loss(loss_fn, observed_outputs, simulated_outputs)[0],
            torch.tensor(0.3333),
            rtol=1e-3,
        )
        simulated_outputs = [
            torch.tensor([1.0, 2, 3]),
            torch.tensor([4.0, 5, float("nan")]),
        ]
        assert compute_loss(loss_fn, observed_outputs, simulated_outputs) == (0, 0)
        simulated_outputs = [
            torch.tensor([1.0, 2, float("nan")]),
            torch.tensor([4.0, 5, float("nan")]),
        ]
        assert np.isnan(compute_loss(loss_fn, observed_outputs, simulated_outputs)[0])

    def test__generate_training_pairs(self):
        def simulator(theta):
            return [theta + torch.randn(1), theta - torch.randn(1), theta + 2 * torch.randn(1)]
        prior = torch.distributions.Normal(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        theta, x = generate_training_data(simulator, prior, 100, progress_bar=False)
        assert theta.shape == torch.Size([100, 2])
        assert x.shape == torch.Size([100, 2, 3])
