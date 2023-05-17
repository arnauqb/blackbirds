import torch
import numpy as np
import warnings
from typing import Callable
from itertools import chain

from birds.mpi_setup import mpi_size, mpi_rank, mpi_comm
from birds.jacfwd import jacfwd


def compute_loss(
    loss_fn: Callable,
    observed_outputs: list[torch.Tensor],
    simulated_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """Compute the loss between observed and simulated outputs.

    **Arguments:**

    - loss_fn : loss function
    - observed_outputs : list of data tensors to calibrate to
    - simulated_outputs: list of simulated outputs

    !!! example
        ```python
        loss_fn = torch.nn.MSELoss()
        observed_outputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        simulated_outputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        compute_loss(loss_fn, observed_outputs, simulated_outputs) # tensor(0.)
        ```
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
    loss_fn: Callable,
    model: torch.nn.Module,
    parameter_generator: Callable,
    n_samples: int,
    observed_outputs: list[torch.Tensor],
    diff_mode: str = "reverse",
    jacobian_chunk_size: int | None = None,
    device: str = "cpu",
):
    """Computes the loss and the jacobian of the loss for each sample.
    The jacobian is computed using the forward or reverse mode differentiation and the computation is parallelized
    across the available devices.

    **Arguments:**

    - `loss_fn`: loss function
    - `model`: PyTorch model
    - `parameter_generator`: parameter generator
    - `n_samples`: number of samples
    - `observed_outputs`: observed outputs
    - `diff_mode`: differentiation mode can be "reverse" or "forward"
    - `jacobian_chunk_size`: chunk size for the Jacobian computation (set None to get maximum chunk size)
    - `device`: device to use for the computation

    !!! example
        ```python
            loss_fn = torch.nn.MSELoss()
            model = lambda x: [x**2]
            parameter_generator = lambda: torch.tensor(2.0)
            observed_outputs = [toch.tensor(4.0)]
            compute_forecast_loss(loss_fn, model, parameter_generator, 5, observed_outputs) # torch.tensor(0.)
        ```
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

    # define loss to differentiate
    def loss_f(params):
        simulated_outputs = model(params)
        loss = compute_loss(loss_fn, observed_outputs, simulated_outputs)
        return loss

    jacobian_calculator = jacobian_diff_mode(
        func=loss_f,
        argnums=0,
        has_aux=True,
        chunk_size=jacobian_chunk_size,
    )
    # make each rank compute the loss for its parameters
    loss = 0
    jacobians_per_rank = []
    indices_per_rank = []  # need to keep track of which parameter has which jacobian
    for i in range(mpi_rank, len(params_list_comm), mpi_size):
        params = torch.tensor(params_list_comm[i], device=device)
        jacobian, loss_i = jacobian_calculator(params)
        if torch.isnan(loss_i) or torch.isnan(jacobian).any():
            continue
        loss += loss_i
        jacobians_per_rank.append(torch.tensor(jacobian.cpu().numpy()))
        indices_per_rank.append(i)
    # gather the jacobians and parameters from all ranks
    if mpi_size > 1:
        jacobians_per_rank = mpi_comm.gather(jacobians_per_rank, root=0)
        indices_per_rank = mpi_comm.gather(indices_per_rank, root=0)
    else:
        jacobians_per_rank = [jacobians_per_rank]
        indices_per_rank = [indices_per_rank]
    if mpi_comm is not None:
        losses = mpi_comm.gather(loss, root=0)
        if mpi_rank == 0:
            loss = sum([l.cpu() for l in losses if l != 0])
    if mpi_rank == 0:
        jacobians = list(chain(*jacobians_per_rank))
        indices = list(chain(*indices_per_rank))
        parameters = params_list[indices]
        loss = loss / len(parameters)
        return parameters, loss, jacobians
    else:
        return None, None, None
