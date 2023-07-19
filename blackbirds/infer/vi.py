import numpy as np
import torch
import logging
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from typing import Callable, List
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from blackbirds.mpi_setup import mpi_rank
from blackbirds.jacfwd import jacfwd
from blackbirds.mpi_setup import mpi_size, mpi_rank, mpi_comm

logger = logging.getLogger("vi")


def compute_regularisation_loss(
    posterior_estimator: torch.nn.Module,
    prior: torch.distributions.Distribution,
    n_samples: int,
):
    r"""Estimates the KL divergence between the posterior and the prior using n_samples through Monte Carlo using

    $$
    \mathbb{E}_{q(z|x)}[\log q(z|x) - \log p(z)] \approx \frac{1}{N} \sum_{i=1}^N \left(\log q(z_i|x) - \log p(z_i)\right)
    $$

    **Arguments**:

    - `posterior_estimator`: The posterior distribution.
    - `prior`: The prior distribution.
    - `n_samples`: The number of samples to use for the Monte Carlo estimate.

    !!! example
        ```python
            import torch
            from blackbirds.regularisation import compute_regularisation
            # define two normal distributions
            dist1 = torch.distributions.Normal(0, 1)
            dist2 = torch.distributions.Normal(0, 1)
            compute_regularisation(dist1, dist2, 1000)
            # tensor(0.)
            dist1 = torch.distributions.Normal(0, 1)
            dist2 = torch.distributions.Normal(1, 1)
            compute_regularisation(dist1, dist2, 1000)
            # tensor(0.5)
        ```
    """
    # sample from the posterior
    z, log_prob_posterior = posterior_estimator.sample(n_samples)
    # compute the log probability of the samples under the prior
    # log_prob_posterior = posterior_estimator.log_prob(z)
    log_prob_prior = prior.log_prob(z)
    # compute the Monte Carlo estimate of the KL divergence
    kl_divergence = (log_prob_posterior - log_prob_prior).mean()
    # kl_divergence = torch.clamp(kl_divergence, min=0.0, max=1)
    return kl_divergence


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


def _differentiate_loss_pathwise(parameters, jacobians):
    """
    Differentiates loss and regularisation loss through the flows and the simulator.

    Arguments:
        parameters (List[torch.Tensor]): The parameters of the simulator that are differentiated through.
        jacobians (List[torch.Tensor]): The jacobians of the simulator that are differentiated through.
    """
    # then we differentiate the parameters through the flow also tkaing into account the jacobians of the simulator
    device = parameters.device
    to_diff = torch.zeros(1, device=device)
    for i in range(len(jacobians)):
        to_diff += torch.matmul(jacobians[i].to(device), parameters[i, :])
    to_diff = to_diff / len(jacobians)
    to_diff.backward()


def compute_loss_and_jacobian_pathwise(
    loss_fn: Callable,
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
    def loss_aux(params):
        loss_v = loss_fn(params, observed_outputs)
        return loss_v, loss_v  # need double return for jacobian calculation.

    jacobian_calculator = jacobian_diff_mode(
        func=loss_aux,
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
        try:
            jacobians_per_rank.append(torch.tensor(jacobian.cpu().numpy()))
        except RuntimeError:
            jacobians_per_rank.append(jacobian.cpu())
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
            if type(loss) == int:
                loss = torch.tensor(loss, device=device)
    if mpi_rank == 0:
        jacobians = list(chain(*jacobians_per_rank))
        indices = list(chain(*indices_per_rank))
        parameters = params_list[indices]
        loss = loss / len(parameters)
        return parameters, loss, jacobians
    else:
        return [], 0.0, []


def compute_and_differentiate_loss_score(
    loss_fn: Callable,
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
        loss_i = loss_fn(params, observed_outputs)
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


def compute_and_differentiate_loss(
    loss_fn: Callable,
    posterior_estimator: torch.nn.Module,
    n_samples: int,
    observed_outputs: list[torch.Tensor],
    diff_mode: str = "reverse",
    gradient_estimation_method: str = "pathwise",
    jacobian_chunk_size: int | None = None,
    gradient_horizon: int = 0,
    device: str = "cpu",
):
    r"""Computes and differentiates the loss according to the chosen gradient estimation method
    and automatic differentiation mechanism.
    **Arguments:**

    - `loss_fn`: loss function
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
            parameters,
            loss,
            jacobians,
        ) = compute_loss_and_jacobian_pathwise(
            loss_fn=loss_fn,
            posterior_estimator=posterior_estimator,
            observed_outputs=observed_outputs,
            n_samples=n_samples,
            diff_mode=diff_mode,
            device=device,
            jacobian_chunk_size=jacobian_chunk_size,
            gradient_horizon=gradient_horizon,
        )
        if mpi_rank == 0:
            _differentiate_loss_pathwise(parameters, jacobians)
    elif gradient_estimation_method == "score":
        loss = compute_and_differentiate_loss_score(
            loss_fn=loss_fn,
            posterior_estimator=posterior_estimator,
            observed_outputs=observed_outputs,
            n_samples=n_samples,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown gradient estimation method {gradient_estimation_method}."
        )
    return loss


class VI:
    """
    Class to handle (Generalized) Variational Inferece.

    **Arguments:**

    - `loss` : A callable that returns a (differentiable) loss. Needs to take (parameters, data) as input and return a scalar tensor.
    - `prior`: The prior distribution.
    - `posterior_estimator`: The variational distribution that approximates the (generalised) posterior.
    - `w`: The weight of the regularisation loss in the total loss.
    - `initialize_estimator_to_prior`: Whether to fit the posterior estimator to the prior before training.
    - `initialization_lr`: The learning rate to use for the initialization.
    - `gradient_clipping_norm`: The norm to which the gradients are clipped.
    - `optimizer`: The optimizer to use for training.
    - `n_samples_per_epoch`: The number of samples to draw from the variational distribution per epoch.
    - `n_samples_regularisation`: The number of samples used to evaluate the regularisation loss.
    - `diff_mode`: The differentiation mode to use. Can be either 'reverse' or 'forward'.
    - `gradient_estimation_method`: The method to use for estimating the gradients of the loss. Can be either 'pathwise' or 'score'.
    - `jacobian_chunk_size` : The number of rows computed at a time for the model Jacobian. Set to None to compute the full Jacobian at once.
    - `gradient_horizon`: The number of timesteps to use for the gradient horizon. Set 0 to use the full trajectory.
    - `device`: The device to use for training.
    - `progress_bar`: Whether to display a progress bar during training.
    - `progress_info` : Whether to display loss data during training.
    - `log_tensorboard`: Whether to log tensorboard data.
    - `tensorboard_log_dir`: The directory to log tensorboard data to.
    """

    def __init__(
        self,
        loss: Callable,
        prior: torch.distributions.Distribution,
        posterior_estimator: torch.nn.Module,
        w: float = 1.0,
        initialize_estimator_to_prior: bool = False,
        initialization_lr: float = 1e-3,
        gradient_clipping_norm: float = np.inf,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        n_samples_per_epoch: int = 10,
        n_samples_regularisation: int = 10_000,
        diff_mode: str = "reverse",
        gradient_estimation_method: str = "pathwise",
        jacobian_chunk_size: int | None = None,
        gradient_horizon: int | float = np.inf,
        device: str = "cpu",
        progress_bar: bool = True,
        progress_info: bool = True,
        log_tensorboard: bool = False,
        tensorboard_log_dir: str | None = None,
    ):
        self.loss = loss
        self.prior = prior
        self.posterior_estimator = posterior_estimator
        self.w = w
        self.initialize_estimator_to_prior = initialize_estimator_to_prior
        self.initialization_lr = initialization_lr
        self.gradient_clipping_norm = gradient_clipping_norm
        if optimizer is None:
            optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=1e-3)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_samples_per_epoch = n_samples_per_epoch
        self.n_samples_regularisation = n_samples_regularisation
        self.progress_bar = progress_bar
        self.progress_info = progress_info
        self.diff_mode = diff_mode
        self.gradient_estimation_method = gradient_estimation_method
        self.jacobian_chunk_size = jacobian_chunk_size
        self.gradient_horizon = gradient_horizon
        self.device = device
        self.tensorboard_log_dir = tensorboard_log_dir
        self.log_tensorboard = log_tensorboard

    def step(self, data):
        """
        Performs one training step.
        """
        if mpi_rank == 0:
            self.optimizer.zero_grad()
        # compute and differentiate loss
        loss = compute_and_differentiate_loss(
            loss_fn=self.loss,
            posterior_estimator=self.posterior_estimator,
            n_samples=self.n_samples_per_epoch,
            observed_outputs=data,
            diff_mode=self.diff_mode,
            gradient_estimation_method=self.gradient_estimation_method,
            jacobian_chunk_size=self.jacobian_chunk_size,
            gradient_horizon=self.gradient_horizon,
            device=self.device,
        )
        # compute and differentiate regularisation loss
        if mpi_rank == 0:
            if self.w != 0.0:
                regularisation_loss = self.w * compute_regularisation_loss(
                    posterior_estimator=self.posterior_estimator,
                    prior=self.prior,
                    n_samples=self.n_samples_regularisation,
                )
                # differentiate regularisation
                regularisation_loss.backward()
            else:
                regularisation_loss = torch.zeros(1, device=loss.device)
            # clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.posterior_estimator.parameters(), self.gradient_clipping_norm
            )
            self.optimizer.step()
            total_loss = loss + regularisation_loss
            return total_loss, loss, regularisation_loss
        return None, None, None

    def initialize_estimator(self, max_epochs_without_improvement=50, atol=1e-2):
        """
        Initialization step where the estimator is fitted to just the prior.
        """
        epoch = 0
        if mpi_rank == 0:
            optimizer = torch.optim.Adam(
                self.posterior_estimator.parameters(), lr=self.initialization_lr
            )
            best_loss = torch.tensor(np.inf)
            while True:
                optimizer.zero_grad()
                loss = compute_regularisation_loss(
                    posterior_estimator=self.posterior_estimator,
                    prior=self.prior,
                    n_samples=self.n_samples_regularisation,
                )
                if self.log_tensorboard:
                    self.writer.add_scalar("Loss/init_loss", loss, epoch)
                loss.backward()
                optimizer.step()
                if loss < best_loss:
                    best_loss = loss.item()
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1
                if (
                    num_epochs_without_improvement >= max_epochs_without_improvement
                    or (loss.abs().item() < atol)
                ):
                    break
                epoch += 1

    def run(
        self,
        data: List[torch.Tensor],
        n_epochs: int,
        max_epochs_without_improvement: int = 20,
    ):
        """
        Runs the calibrator for {n_epochs} epochs. Stops if the loss does not improve for {max_epochs_without_improvement} epochs.

        **Arguments:**

        - `data`: The observed data to calibrate against. It must be given as a list of tensors that matches the output of the model.
        - `n_epochs`: The number of epochs to run the calibrator for.
        - `max_epochs_without_improvement`: The number of epochs without improvement after which the calibrator stops.
        """
        if mpi_rank == 0 and self.log_tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        if self.initialize_estimator_to_prior:
            self.initialize_estimator()
            torch.save(
                self.posterior_estimator.state_dict(), "estimator_fit_to_prior.pt"
            )
        self.best_loss = torch.tensor(np.inf)
        self.best_estimator_state_dict = None
        num_epochs_without_improvement = 0
        iterator = range(n_epochs)
        if self.progress_bar and mpi_rank == 0:
            iterator = tqdm(iterator)
        self.losses_hist = defaultdict(list)
        for epoch in iterator:
            total_loss, loss, regularisation_loss = self.step(data)
            if mpi_rank == 0:
                self.losses_hist["total"].append(total_loss.item())
                self.losses_hist["loss"].append(loss.item())
                self.losses_hist["regularisation"].append(regularisation_loss.item())
                if self.log_tensorboard:
                    self.writer.add_scalar("Loss/total", total_loss, epoch)
                    self.writer.add_scalar("Loss/loss", loss, epoch)
                    self.writer.add_scalar(
                        "Loss/regularisation", regularisation_loss, epoch
                    )
                torch.save(self.best_estimator_state_dict, "last_estimator.pt")
                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    self.best_estimator_state_dict = deepcopy(
                        self.posterior_estimator.state_dict()
                    )
                    torch.save(self.best_estimator_state_dict, "best_estimator.pt")
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1
                if self.progress_bar and self.progress_info:
                    iterator.set_postfix(
                        {
                            "loss": loss.item(),
                            "reg.": regularisation_loss.item(),
                            "total": total_loss.item(),
                            "best loss": self.best_loss.item(),
                            "epochs since improv.": num_epochs_without_improvement,
                        }
                    )
                if num_epochs_without_improvement >= max_epochs_without_improvement:
                    logger.info(
                        "Stopping early because the loss did not improve for {} epochs.".format(
                            max_epochs_without_improvement
                        )
                    )
                    break
            if not self.scheduler is None:
                self.scheduler.step(total_loss)
        if mpi_rank == 0 and self.log_tensorboard:
            self.writer.flush()
            self.writer.close()
