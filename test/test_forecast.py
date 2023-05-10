import pytest
import numpy as np
import torch

from birds.forecast import compute_loss


def test__compute_loss():
    r"""Test compute_loss function."""
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
