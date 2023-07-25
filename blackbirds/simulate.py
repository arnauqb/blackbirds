import torch
import warnings
from tqdm import tqdm
from typing import Callable

from blackbirds.mpi_setup import mpi_rank, mpi_size, mpi_comm


def simulate_and_observe_model(
    model: torch.nn.Module,
    params: torch.Tensor,
    gradient_horizon: int | None = None,
):
    """Runs the simulator for the given parameters and calls the model's observe method.
    To avoid gradient instabilities, the `gradient_horizon` argument limits the number of past time-steps
    that are taken into account for the gradient's calculation. That is, if `gradient_horizon` is 10, then
    only the last 10 time-steps are used to calculate the gradient.

    **Arguments:**

    - `model`: A torch.nn.Module implemnting the `initialize`, `forward` and `observe` methods.
    - `params`: The parameters taken by the model's `forward` method.
    - `n_timesteps`: Number of timesteps to simulate.
    - `gradient_horizon`: Gradient window, if None then all time-steps are used to calculate the gradient.
    """
    if gradient_horizon is None:
        gradient_horizon = model.n_timesteps
    # Initialize the model
    time_series = model.initialize(params)
    observed_outputs = model.observe(time_series)
    for t in range(model.n_timesteps):
        time_series = model.trim_time_series(
            time_series
        )  # gets past time-steps needed to compute the next one.
        # only consider the past gradient_horizon time-steps to calculate the gradient
        if t > gradient_horizon:
            time_series = model.detach_gradient_horizon(time_series, gradient_horizon)
        x = model(params, time_series)
        observed_outputs = [
            torch.cat((observed_output, output))
            for observed_output, output in zip(observed_outputs, model.observe(x))
        ]
        if time_series is not None:
            time_series = torch.cat((time_series, x))
    return observed_outputs


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
        raise ValueError("Number of observed and simulated outputs must be the same.")
    loss = 0
    is_nan = True
    for observed_output, simulated_output in zip(observed_outputs, simulated_outputs):
        try:
            assert observed_output.shape == simulated_output.shape
        except AssertionError:
            raise ValueError("Observed and simulated outputs must have the same shape")
        if torch.isnan(simulated_output).any():
            warnings.warn("Simulation produced nan -- ignoring")
            continue
        loss += loss_fn(simulated_output, observed_output)
        is_nan = False
    if is_nan:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)
    return loss, loss  # need to return it twice for jac calculation


def generate_training_data(
    simulator: Callable,
    prior: torch.distributions.Distribution,
    n_training_samples: int = 1_000,
    progress_bar = True
    ):
    """
    Generates training data pairs (theta, x) where theta are the 
    model parameters and x the model output. Supports MPI parallelilsation
    for multi-gpu calculation.

    **Arguments:**

    - `simulator`: A function that takes a parameter vector theta and returns a model output x.
    - `prior`: A torch.distributions.Distribution object that generates training parameter samples.
    - `n_training_samples`: Number of training samples to generate.

    !!! example
        ```python
        def simulator(theta):
            return theta + torch.randn(1)
        prior = torch.distributions.Normal(0., 1.)
        theta, x = generate_training_data(simulator, prior, 1000)
        ```
    To run in parallel, simply run the following command:
        ```bash
        mpirun -np X python script.py
        ```
    where `script.py` contains the code in the example above.
    """
    # sample and scatter parameters across mpi ranks
    thetas = prior.sample((n_training_samples,))
    thetas = torch.split(thetas, n_training_samples // mpi_size)[mpi_rank]
    # simulate
    xs = []
    if progress_bar:
        prange = tqdm(range(len(thetas)))
    else:
        prange = range(len(thetas))
    for i in prange:
        theta = thetas[i]
        x = simulator(theta)
        x = torch.cat([t[:,None] for t in x], dim=-1)
        xs.append(x[None,:])
    # concatenate xs into tensor for size N x T x F
    # where N is number of samples, T is number of timesteps and F is number of features
    xs 
    xs = torch.cat(xs, dim=0)
    # gather all parameters and outputs to rank 0
    if mpi_comm is not None and mpi_size > 1:
        thetas = mpi_comm.gather(thetas, root=0)
        xs = mpi_comm.gather(xs, root=0)
        if mpi_rank == 0:
            thetas = torch.cat(thetas, dim=0)
            xs = torch.cat(xs, dim=0)
    if mpi_rank == 0:
        assert thetas.shape[0] == n_training_samples
        assert xs.shape[0] == n_training_samples
        return thetas, xs
    else:
        return [], []

