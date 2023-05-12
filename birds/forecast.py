import torch
import numpy as np
import warnings
from itertools import chain

from birds.mpi_setup import mpi_size, mpi_rank, mpi_comm
from birds.jacfwd import jacfwd


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
        return torch.nan, torch.nan
    return loss, loss  # need to return it twice for jac calculation


def compute_forecast_loss_and_jacobian(
    loss_fn,
    model,
    parameter_generator,
    n_samples,
    observed_outputs,
    diff_mode="reverse",
    jacobian_chunk_size=None,
    device="cpu",
):
    r"""Computes the loss and the jacobian of the loss for each sample.
    The jacobian is computed using the forward or reverse mode differentiation and the computation is parallelized
    across the available devices.
    Arguments:
        loss_fn (callable) : loss function
        model (callable) : PyTorch model
        parameter_generator (callable) : parameter generator
        n_samples (int) : number of samples
        observed_outputs (list of torch.Tensor) : observed outputs
        diff_mode (str) : differentiation mode can be "reverse" or "forward"
        jacobian_chunk_size (int) : chunk size for the Jacobian computation (set None to get maximum chunk size)
        device (str) : device to use for the computation
    Example:
        >>> loss_fn = torch.nn.MSELoss()
        >>> model = lambda x: [x**2]
        >>> parameter_generator = lambda: torch.tensor(2.0)
        >>> observed_outputs = [toch.tensor(4.0)]
        >>> compute_forecast_loss(loss_fn, model, parameter_generator, 5, observed_outputs)
        tensor(0.)
    """
    # Rank 0 samples from the flow
    if mpi_rank == 0:
        params_list = parameter_generator(n_samples)
        params_list_comm = params_list.detach().cpu().numpy()
    else:
        params_list_comm = None
    # scatter the parameters to all ranks
    if mpi_comm is not None:
        params_list_comm = mpi_comm.bcast(params_list_comm, root=0)
    # select forward or reverse jacobian calculator
    if diff_mode == "reverse":
        jacobian_diff_mode = torch.func.jacrev
    else:
        jacobian_diff_mode = lambda **kwargs: jacfwd(randomness="same", **kwargs)
    loss_f = lambda x: compute_loss(
        loss_fn=loss_fn, observed_outputs=observed_outputs, simulated_outputs=x
    )
    jacobian_calculator = jacobian_diff_mode(
        func=loss_f,
        argnums=0,
        has_aux=True,
        chunk_size=jacobian_chunk_size,
    )
    # make each rank compute the loss for its parameters
    loss = 0
    jacobians_per_rank = []
    indices_per_rank = []
    for i in range(mpi_rank, len(params_list_comm), mpi_size):
        params = params_list_comm[i]
        simulated_outputs = model(torch.tensor(params, device=device))
        jacobian, loss_i = jacobian_calculator(simulated_outputs)
        if np.isnan(loss):
            continue
        loss += loss_i
        jacobians_per_rank.append(torch.tensor(jacobian[0].cpu().numpy()))
        indices_per_rank.append(i)
    # gather the jacobians and parameters from all ranks
    if mpi_size > 1:
        jacobians_per_rank = mpi_comm.gather(jacobians_per_rank, root=0)
        indices_per_rank = mpi_comm.gather(indices_per_rank, root=0)
    else:
        jacobians_per_rank = [jacobians_per_rank]
        indices_per_rank = [indices_per_rank]
    if mpi_rank == 0:
        jacobians = []
        jacobians = list(chain(*jacobians_per_rank))
        if mpi_comm is not None:
            loss = sum(mpi_comm.gather(loss, root=0))
    if mpi_rank == 0:
        indices = list(chain(*indices_per_rank))
        parameters = params_list[indices]
        loss = loss / len(parameters)
        return parameters, loss, jacobians
    else:
        return None, None, None
