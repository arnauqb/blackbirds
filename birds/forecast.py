import torch
import numpy as np
import warnings


def compute_loss(loss_fn, observed_outputs, simulated_outputs):
    r"""Compute the loss between observed and simulated outputs.

    Arguments:
        loss_fn : callable
        observed_outputs : list of torch.Tensor
        simulated_outputs : list of torch.Tensor

    Example:
        >>> loss_fn = torch.nn.MSELoss()
        >>> observed_outputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> simulated_outputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> compute_loss(loss_fn, observed_outputs, simulated_outputs)
        tensor(0.)
    """
    try:
        assert len(observed_outputs) == len(simulated_outputs)
    except AssertionError:
        raise ValueError("Observed and simulated outputs must be the same length")
    loss = 0
    is_nan = True
    for observed_output, simulated_output in zip(observed_outputs, simulated_outputs):
        if torch.isnan(simulated_output).any():
            warnings.warn("Simulation produced nan -- ignoring")
            continue
        loss += loss_fn(observed_output, simulated_output)
        is_nan = False
    if is_nan:
        return torch.nan
    return loss


def compute_forecast_loss(
    loss_fn, model, parameter_generator, n_samples, observed_outputs
):
    r"""Given a model and a parameter generator, compute the loss between the model outputs and the observed outputs.

    Arguments:
        loss_fn : callable
        model : callable
        parameter_generator : callable
        n_samples : int
        observed_outputs : list of torch.Tensor
    Example:
        >>> loss_fn = torch.nn.MSELoss()
        >>> model = lambda x: [x**2]
        >>> parameter_generator = lambda: torch.tensor(2.0)
        >>> observed_outputs = [toch.tensor(4.0)]
        >>> compute_forecast_loss(loss_fn, model, parameter_generator, 5, observed_outputs)
        tensor(0.)
    """
    n_samples_not_nan = 0
    loss = 0
    for _ in range(n_samples):
        parameters = parameter_generator()
        simulated_outputs = model(parameters)
        loss_i = compute_loss(loss_fn, observed_outputs, simulated_outputs)
        if np.isnan(loss_i):
            continue
        loss += loss_i
        n_samples_not_nan += 1
    if n_samples_not_nan == 0:
        loss = torch.nan
    else:
        loss = loss / n_samples_not_nan
    return loss, loss # need to return it twice for the jacobian calculation

