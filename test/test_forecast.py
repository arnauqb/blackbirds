import numpy as np
import torch
import pytest

from birds.forecast import (
    compute_loss,
    compute_forecast_loss_and_jacobian_pathwise,
    compute_and_differentiate_forecast_loss,
)

class MockPosteriorEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.tensor(2.0, requires_grad=True)

    def sample(self, n):
        # Needs to return sample and log prob of the sample.
        x = self.p * torch.ones((n, 2))
        log_prob = (x * 1/n).sum() * torch.ones(n)
        return x, log_prob

    def forward(self, n):
        return self.sample(n)

class MockModel(torch.nn.Module):
    def forward(self, x):
        return [x**2]

@pytest.fixture(name="mock_estimator")
def make_mock_posterior():
    return MockPosteriorEstimator()

@pytest.fixture(name="mock_model")
def make_mock_model():
    return MockModel()


class TestForecast:
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
        observed_outputs = [torch.tensor([4.0, 4.0])]
        parameters, loss, jacobians = compute_forecast_loss_and_jacobian_pathwise(
            loss_fn, mock_model, mock_estimator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, torch.tensor([2.0, 2.0]))
        assert loss == torch.tensor(0.0)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([0.0, 0.0]))

        observed_outputs = [torch.tensor([2.0, 3.0])]
        parameters, loss, jacobians = compute_forecast_loss_and_jacobian_pathwise(
            loss_fn, mock_model, mock_estimator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, torch.tensor([2.0, 2.0]))
        assert loss == torch.tensor(2.5)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([8.0, 4.0]))

    @pytest.mark.parametrize("diff_mode", ["forward", "reverse"])
    @pytest.mark.parametrize("gradient_estimation_method", ["score", "pathwise"])
    def test__compute_forecast_loss(self, mock_estimator, mock_model, diff_mode, gradient_estimation_method):
        loss_fn = torch.nn.MSELoss()
        observed_outputs = [torch.tensor([4.0, 4.0])]
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
        observed_outputs = [torch.tensor([2.0, 3.0])]
        forecast_loss = compute_and_differentiate_forecast_loss(
            loss_fn,
            mock_model,
            mock_estimator,
            5,
            observed_outputs,
            diff_mode=diff_mode,
            gradient_estimation_method=gradient_estimation_method,
        )
        assert np.isclose(forecast_loss, 2.5)
