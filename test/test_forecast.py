import numpy as np
import torch

from birds.forecast import compute_loss, compute_forecast_loss_and_jacobian

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

    def test__compute_forecast_loss(self):
        loss_fn = torch.nn.MSELoss()
        model = lambda x: [x ** 2]
        parameter_generator = lambda x: 2.0 * torch.ones((x, 2))
        observed_outputs = [torch.tensor([4.0, 4.0])]
        parameters, loss, jacobians = compute_forecast_loss_and_jacobian(
            loss_fn, model, parameter_generator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, torch.tensor([2.0, 2.0]))
        assert loss == torch.tensor(0.0)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([0.0, 0.0]))

