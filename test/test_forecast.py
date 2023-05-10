import pytest
import numpy as np
import torch

from birds.forecast import compute_loss, compute_forecast_loss


class TestForecast:
    def test__compute_loss(self):
        loss_fn = torch.nn.MSELoss()
        observed_outputs = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6])]
        simulated_outputs = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6])]
        assert compute_loss(loss_fn, observed_outputs, simulated_outputs) == 0
        simulated_outputs = [torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 7])]
        assert torch.isclose(
            compute_loss(loss_fn, observed_outputs, simulated_outputs),
            torch.tensor(0.3333),
            rtol=1e-3,
        )
        simulated_outputs = [
            torch.tensor([1.0, 2, 3]),
            torch.tensor([4.0, 5, float("nan")]),
        ]
        assert compute_loss(loss_fn, observed_outputs, simulated_outputs) == 0
        simulated_outputs = [
            torch.tensor([1.0, 2, float("nan")]),
            torch.tensor([4.0, 5, float("nan")]),
        ]
        assert np.isnan(compute_loss(loss_fn, observed_outputs, simulated_outputs))

    def test__compute_forecast_loss(self):
        loss_fn = torch.nn.MSELoss()
        model = lambda x: [x ** 2]
        parameter_generator = lambda: torch.tensor(2.0)
        observed_outputs = [torch.tensor(4.0)]
        assert compute_forecast_loss(
            loss_fn, model, parameter_generator, 5, observed_outputs
        ) == 0
        parameter_generator = lambda: torch.tensor(float("nan"))
        assert np.isnan(
            compute_forecast_loss(
                loss_fn, model, parameter_generator, 5, observed_outputs
            )
        )
        parameter_generator = lambda: torch.tensor(2.0)
        model = lambda x: [x ** 3]
        assert compute_forecast_loss(
            loss_fn, model, parameter_generator, 5, observed_outputs
        ) == (8-4)**2
