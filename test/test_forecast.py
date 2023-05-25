import numpy as np
import torch
import pytest

from birds.forecast import (
    simulate_and_observe_model,
    compute_loss,
    compute_forecast_loss_and_jacobian_pathwise,
    compute_and_differentiate_forecast_loss,
)
from birds.models.model import Model


class MockPosteriorEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.tensor(1.0, requires_grad=True)

    def sample(self, n):
        x = 2 * self.p * torch.ones((n, 2))
        return x, self.log_prob(x)

    def log_prob(self, x):
        if len(x.shape) == 1:
            return torch.zeros(1) * self.p
        else:
            return torch.zeros(x.shape[0]) * self.p

    def forward(self, n):
        return self.sample(n)


class MockModel(Model):
    def __init__(self):
        super().__init__()
        self.n_timesteps = 2

    def trim_time_series(self, x):
        return x[-1:]

    def initialize(self, params):
        x = (params**2).reshape(1, -1)
        return x

    def step(self, params, x):
        return (params**2).reshape(1, -1)

    def observe(self, x):
        return [x]


@pytest.fixture(name="mock_estimator")
def make_mock_posterior():
    return MockPosteriorEstimator()


@pytest.fixture(name="mock_model")
def make_mock_model():
    return MockModel()


class TestForecast:
    def test__simulate_and_observe_model(self, mock_model):
        params = torch.tensor([2.0], requires_grad=True)
        observations = simulate_and_observe_model(
            model=mock_model, params=params, gradient_horizon=0
        )
        assert len(observations) == 1
        assert (observations[0] == torch.tensor([4.0, 4.0, 4.0])).all()
        y = torch.ones(3,1)
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

    def test__compute_forecast_loss_pathwise(self, mock_estimator, mock_model):
        loss_fn = torch.nn.MSELoss()
        observed_outputs = [4 * torch.ones(3, 2)]
        parameters, loss, jacobians = compute_forecast_loss_and_jacobian_pathwise(
            loss_fn, mock_model, mock_estimator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, 2.0 * torch.ones(2))
        assert loss == torch.tensor(0.0)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([0.0, 0.0]))

        observed_outputs = [
            torch.tensor([2.0, 3.0, 4.0]).reshape(-1, 1) * torch.ones(3, 2)
        ]
        parameters, loss, jacobians = compute_forecast_loss_and_jacobian_pathwise(
            loss_fn, mock_model, mock_estimator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, torch.tensor([2.0, 2.0]))
        assert loss == torch.tensor(5 / 3)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([4.0, 4.0]))

    @pytest.mark.parametrize("diff_mode", ["forward", "reverse"])
    @pytest.mark.parametrize("gradient_estimation_method", ["score", "pathwise"])
    def test__compute_forecast_loss(
        self, mock_estimator, mock_model, diff_mode, gradient_estimation_method
    ):
        loss_fn = torch.nn.MSELoss()
        observed_outputs = [4 * torch.ones(3, 2)]
        forecast_loss = compute_and_differentiate_forecast_loss(
            loss_fn,
            mock_model,
            mock_estimator,
            5,
            observed_outputs,
            diff_mode=diff_mode,
            gradient_estimation_method=gradient_estimation_method,
        )
        assert np.isclose(forecast_loss, 0.0)
        observed_outputs = [
            torch.tensor([2.0, 3.0, 4.0]).reshape(-1, 1) * torch.ones(3, 2)
        ]
        forecast_loss = compute_and_differentiate_forecast_loss(
            loss_fn,
            mock_model,
            mock_estimator,
            5,
            observed_outputs,
            diff_mode=diff_mode,
            gradient_estimation_method=gradient_estimation_method,
        )
        assert np.isclose(forecast_loss, 5 / 3)
        if gradient_estimation_method == "pathwise":
            # for score since the mock one is constant this is 0...
            assert np.isclose(mock_estimator.p.grad.item(), 16, rtol=1e-3)
