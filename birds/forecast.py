import torch
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
