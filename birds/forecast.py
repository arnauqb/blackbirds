import torch
import numpy as np
import warnings
from typing import Callable
from itertools import chain

from birds.mpi_setup import mpi_size, mpi_rank, mpi_comm
from birds.jacfwd import jacfwd


def simulate_and_observe_model(
    model: torch.nn.Module,
    params: torch.Tensor,
    gradient_horizon: int = 0,
):
    """Runs the simulator for the given parameters and calls the model's observe method.
    To avoid gradient instabilities, the `gradient_horizon` argument limits the number of past time-steps
    that are taken into account for the gradient's calculation. That is, if `gradient_horizon` is 10, then
    only the last 10 time-steps are used to calculate the gradient.

    **Arguments:**

    - `model`: A torch.nn.Module implemnting the `initialize`, `forward` and `observe` methods.
    - `params`: The parameters taken by the model's `forward` method.
    - `n_timesteps`: Number of timesteps to simulate.
    - `gradient_horizon`: Gradient window, if 0 then all time-steps are used to calculate the gradient.
    """
    # Initialize the model
    time_series = model.initialize(params)
    observed_outputs = model.observe(time_series)
    for t in range(model.n_timesteps):
        time_series = model.trim_time_series(
            time_series
        )  # gets past time-steps needed to compute the next one.
        if (gradient_horizon != 0) and ((t + 1) % gradient_horizon == 0):
            # reset the gradient
            x = model(params, time_series.detach())
        else:
            x = model(params, time_series)
        observed_outputs = [
            torch.cat((observed_output, output))
            for observed_output, output in zip(observed_outputs, model.observe(x))
        ]
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
        print("observed_output", observed_output)
        print("simulated_output", simulated_output)
        loss += loss_fn(simulated_output, observed_output)
        is_nan = False
    if is_nan:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)
    return loss, loss  # need to return it twice for jac calculation


def _sample_and_scatter_parameters(
    posterior_estimator: Callable,
    n_samples: int,
) -> (torch.Tensor, np.ndarray):
    """Sample parameters and scatter them across devices.

    **Arguments:**

    - posterior_estimator : function that generates parameters
    - n_samples : number of samples to generate

    **Returns:**
    - If rank 0, returns a tensor of parameters and a numpy array of parameters. The first tensor carries the gradient.
    - If not rank 0, returns None and a numpy array of parameters.
    """
    # Rank 0 samples from the flow
    if mpi_rank == 0:
        params_list, logprobs_list = posterior_estimator.sample(n_samples)
        params_list_comm = params_list.detach().cpu().numpy()
    else:
        params_list = None
        params_list_comm = None
        logprobs_list = None
    # scatter the parameters to all ranks
    if mpi_comm is not None:
        params_list_comm = mpi_comm.bcast(params_list_comm, root=0)
    return params_list, params_list_comm, logprobs_list


def _differentiate_forecast_loss_pathwise(forecast_parameters, forecast_jacobians):
    """
    Differentiates forecast loss and regularisation loss through the flows and the simulator.

    Arguments:
        forecast_parameters (List[torch.Tensor]): The parameters of the simulator that are differentiated through.
        forecast_jacobians (List[torch.Tensor]): The jacobians of the simulator that are differentiated through.
    """
    # then we differentiate the parameters through the flow also tkaing into account the jacobians of the simulator
    device = forecast_parameters.device
    to_diff = torch.zeros(1, device=device)
    for i in range(len(forecast_jacobians)):
        to_diff += torch.dot(
            forecast_jacobians[i].to(device), forecast_parameters[i, :]
        )
    to_diff = to_diff / len(forecast_jacobians)
    to_diff.backward()


def compute_forecast_loss_and_jacobian_pathwise(
    loss_fn: Callable,
    model: torch.nn.Module,
    posterior_estimator: Callable,
    n_samples: int,
    observed_outputs: list[torch.Tensor],
    diff_mode: str = "reverse",
    jacobian_chunk_size: int | None = None,
    gradient_horizon: int = 0,
    device: str = "cpu",
):
    r"""Computes the loss and the jacobian of the loss for each sample using a differentiable simulator. That is, we compute

    $$
    \eta = \nabla_\psi \mathbb{E}_{p(\theta | \psi)} \left[ \mathcal{L}(\theta) \right],
    $$

    by performing the pathwise gradient (reparameterization trick),

    $$
    \eta \approx \frac{1}{N} \sum_{i=1}^N \nabla_\psi \mathcal{L}(\theta_i(\psi)).
    $$

    The jacobian is computed using the forward or reverse mode differentiation and the computation is parallelized
    across the available devices.

    **Arguments:**

    - `loss_fn`: loss function
    - `model`: PyTorch model
    - `posterior_estimator`: Object that implements the `sample` method computing a parameter and its log_prob
    - `n_samples`: number of samples
    - `observed_outputs`: observed outputs
    - `diff_mode`: differentiation mode can be "reverse" or "forward"
    - `jacobian_chunk_size`: chunk size for the Jacobian computation (set None to get maximum chunk size)
    - `gradient_horizon`: horizon for the gradient computation
    - `device`: device to use for the computation
    """
    # sample parameters and scatter them across devices
    params_list, params_list_comm, _ = _sample_and_scatter_parameters(
        posterior_estimator, n_samples
    )
    # select forward or reverse jacobian calculator
    if diff_mode == "reverse":
        jacobian_diff_mode = torch.func.jacrev
    else:
        jacobian_diff_mode = lambda **kwargs: jacfwd(randomness="same", **kwargs)

    # define loss to differentiate
    def loss_f(params):
        simulated_outputs = simulate_and_observe_model(model, params, gradient_horizon)
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
        return [], 0.0, []


def compute_and_differentiate_forecast_loss_score(
    loss_fn: Callable,
    model: torch.nn.Module,
    posterior_estimator: torch.nn.Module,
    n_samples: int,
    observed_outputs: list[torch.Tensor],
    device: str = "cpu",
):
    r"""Computes the loss and the jacobian of the loss for each sample using a differentiable simulator. That is, we compute

    $$
    \eta = \nabla_\psi \mathbb{E}_{\psi \sim p(\theta)} \left[ \mathcal{L}(\theta) \right],
    $$

    by performing the score gradient

    $$
    \eta \approx \frac{1}{N} \sum_{i=1}^N \mathcal{L}(\theta_i) \nabla_\psi \log p\left(\theta_i | \psi\right).
    $$

    The jacobian is computed using the forward or reverse mode differentiation and the computation is parallelized
    across the available devices.

    **Arguments:**

    - `loss_fn`: loss function
    - `model`: PyTorch model
    - `posterior_estimator`: posterior estimator, must implement a sample and a log_prob method
    - `n_samples`: number of samples
    - `observed_outputs`: observed outputs
    - `device`: device to use for the computation
    """
    # sample parameters and scatter them across devices
    _, params_list_comm, logprobs_list = _sample_and_scatter_parameters(
        posterior_estimator, n_samples
    )
    # make each rank compute the loss for its parameters
    loss_per_parameter = []
    indices_per_rank = []  # need to keep track of which parameter has which loss
    for i in range(mpi_rank, len(params_list_comm), mpi_size):
        params = torch.tensor(params_list_comm[i], device=device)
        simulated_outputs = simulate_and_observe_model(
            model, params, gradient_horizon=0
        )
        loss_i, _ = compute_loss(loss_fn, observed_outputs, simulated_outputs)
        loss_per_parameter.append(loss_i.detach().cpu().numpy())
        indices_per_rank.append(i)
    # gather the losses from all ranks
    if mpi_size > 1:
        loss_per_parameter = mpi_comm.gather(loss_per_parameter, root=0)
        indices_per_rank = mpi_comm.gather(indices_per_rank, root=0)
    else:
        loss_per_parameter = [loss_per_parameter]
        indices_per_rank = [indices_per_rank]
    # compute the loss times the logprob of each parameter in rank 0
    if mpi_rank == 0:
        loss_per_parameter = list(chain(*loss_per_parameter))
        indices = list(chain(*indices_per_rank))
        logprobs_list = logprobs_list[indices]
        params_list_comm = params_list_comm[indices]
        to_backprop = 0.0
        total_loss = 0.0
        n_samples_non_nan = 0
        for param, loss_i, param_logprob in zip(
            params_list_comm, loss_per_parameter, logprobs_list
        ):
            loss_i = torch.tensor(loss_i, device=device)
            if np.isnan(loss_i):  # no parameter was non-nan
                continue
            lp = posterior_estimator.log_prob(torch.tensor(param.reshape(1, -1)))
            to_backprop += loss_i * lp
            total_loss += loss_i
            n_samples_non_nan += 1
        to_backprop = to_backprop / n_samples_non_nan
        total_loss = total_loss / n_samples_non_nan
        # differentiate through the posterior estimator
        to_backprop.backward()
        return total_loss
    return None


def compute_and_differentiate_forecast_loss(
    loss_fn: Callable,
    model: torch.nn.Module,
    posterior_estimator: torch.nn.Module,
    n_samples: int,
    observed_outputs: list[torch.Tensor],
    diff_mode: str = "reverse",
    gradient_estimation_method: str = "pathwise",
    jacobian_chunk_size: int | None = None,
    gradient_horizon: int = 0,
    device: str = "cpu",
):
    r"""Computes and differentiates the forecast loss according to the chosen gradient estimation method
    and automatic differentiation mechanism.
    **Arguments:**

    - `loss_fn`: loss function
    - `model`: PyTorch model
    - `posterior_estimator`: posterior estimator, must implement a sample and a log_prob method
    - `n_samples`: number of samples
    - `observed_outputs`: observed outputs
    - `diff_mode`: differentiation mode can be "reverse" or "forward"
    - `gradient_estimation_method`: gradient estimation method can be "pathwise" or "score"
    - `jacobian_chunk_size`: chunk size for the Jacobian computation (set None to get maximum chunk size)
    - `gradient_horizon`: horizon for the gradient computation
    - `device`: device to use for the computation
    """
    if gradient_estimation_method == "pathwise":
        (
            forecast_parameters,
            forecast_loss,
            forecast_jacobians,
        ) = compute_forecast_loss_and_jacobian_pathwise(
            loss_fn=loss_fn,
            model=model,
            posterior_estimator=posterior_estimator,
            observed_outputs=observed_outputs,
            n_samples=n_samples,
            diff_mode=diff_mode,
            device=device,
            jacobian_chunk_size=jacobian_chunk_size,
            gradient_horizon=gradient_horizon,
        )
        if mpi_rank == 0:
            _differentiate_forecast_loss_pathwise(
                forecast_parameters, forecast_jacobians
            )
    elif gradient_estimation_method == "score":
        forecast_loss = compute_and_differentiate_forecast_loss_score(
            loss_fn=loss_fn,
            model=model,
            posterior_estimator=posterior_estimator,
            observed_outputs=observed_outputs,
            n_samples=n_samples,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown gradient estimation method {gradient_estimation_method}."
        )
    return forecast_loss
