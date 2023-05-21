import numpy as np
from copy import deepcopy
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from collections import defaultdict
from typing import Callable, List

from birds.mpi_setup import mpi_rank
from birds.models.model import Model
from birds.regularisation import compute_regularisation_loss
from birds.forecast import compute_and_differentiate_forecast_loss

logger = logging.getLogger("calibrator")


class Calibrator:
    """
    Class that handles the training of the posterior_estimator given the model, data, and prior.

    **Arguments:**

    - `model`: The simulator model.
    - `prior`: The prior distribution.
    - `posterior_estimator`: The variational distribution that approximates the generalised posterior.
    - `data`: The observed data to calibrate against. It must be given as a list of tensors that matches the output of the model.
    - `w`: The weight of the regularisation loss in the total loss.
    - `gradient_clipping_norm`: The norm to which the gradients are clipped.
    - `forecast_loss`: The loss function to use for the forecast loss.
    - `optimizer`: The optimizer to use for training.
    - `n_samples_per_epoch`: The number of samples to draw from the variational distribution per epoch.
    - `n_samples_regularisation`: The number of samples used to evaluate the regularisation loss.
    - `diff_mode`: The differentiation mode to use. Can be either 'reverse' or 'forward'.
    - `gradient_estimation_method`: The method to use for estimating the gradients of the forecast loss. Can be either 'pathwise' or 'score'.
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
        model: Model,
        prior: torch.distributions.Distribution,
        posterior_estimator: torch.nn.Module,
        data: List[torch.Tensor],
        w: float = 0.0,
        gradient_clipping_norm: float = np.inf,
        forecast_loss: Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        n_samples_per_epoch: int = 5,
        n_samples_regularisation: int = 10_000,
        diff_mode: str = "reverse",
        gradient_estimation_method: str = "pathwise",
        jacobian_chunk_size: int | None = None,
        gradient_horizon: int = 0,
        device: str = "cpu",
        progress_bar: bool = True,
        progress_info: bool = False,
        log_tensorboard: bool = False,
        tensorboard_log_dir: str | None = None,
    ):
        self.model = model
        self.prior = prior
        self.posterior_estimator = posterior_estimator
        self.data = data
        self.w = w
        self.gradient_clipping_norm = gradient_clipping_norm
        if forecast_loss is None:
            forecast_loss = torch.nn.MSELoss()
        self.forecast_loss = forecast_loss
        if optimizer is None:
            optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=1e-3)
        self.optimizer = optimizer
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

    def step(self):
        """
        Performs one training step.
        """
        if mpi_rank == 0:
            self.optimizer.zero_grad()
        # compute and differentiate forecast loss
        forecast_loss = compute_and_differentiate_forecast_loss(
            loss_fn=self.forecast_loss,
            model=self.model,
            posterior_estimator=self.posterior_estimator,
            n_samples=self.n_samples_per_epoch,
            observed_outputs=self.data,
            diff_mode=self.diff_mode,
            gradient_estimation_method=self.gradient_estimation_method,
            jacobian_chunk_size=self.jacobian_chunk_size,
            gradient_horizon=self.gradient_horizon,
            device=self.device,
        )
        # compute and differentiate regularisation loss
        if mpi_rank == 0:
            regularisation_loss = self.w * compute_regularisation_loss(
                posterior_estimator=self.posterior_estimator,
                prior=self.prior,
                n_samples=self.n_samples_regularisation,
            )
            # differentiate regularisation
            regularisation_loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.posterior_estimator.parameters(), self.gradient_clipping_norm
            )
            self.optimizer.step()
            loss = forecast_loss + regularisation_loss
            return loss, forecast_loss, regularisation_loss
        return None, None, None

    def run(self, n_epochs, max_epochs_without_improvement=20):
        """
        Runs the calibrator for {n_epochs} epochs. Stops if the loss does not improve for {max_epochs_without_improvement} epochs.

        Arguments:
            n_epochs (int | np.inf): The number of epochs to run the calibrator for.
            max_epochs_without_improvement (int): The number of epochs without improvement after which the calibrator stops.
        """
        self.best_loss = torch.tensor(np.inf)
        self.best_model_state_dict = None
        if mpi_rank == 0 and self.log_tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        num_epochs_without_improvement = 0
        iterator = range(n_epochs)
        if self.progress_bar and mpi_rank == 0:
            iterator = tqdm(iterator)
        self.losses_hist = defaultdict(list)
        for epoch in iterator:
            loss, forecast_loss, regularisation_loss = self.step()
            if mpi_rank == 0:
                self.losses_hist["total"].append(loss.item())
                self.losses_hist["forecast"].append(forecast_loss.item())
                self.losses_hist["regularisation"].append(regularisation_loss.item())
                if self.log_tensorboard:
                    self.writer.add_scalar("Loss/total", loss, epoch)
                    self.writer.add_scalar("Loss/forecast", forecast_loss, epoch)
                    self.writer.add_scalar(
                        "Loss/regularisation", regularisation_loss, epoch
                    )
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_model_state_dict = deepcopy(
                        self.posterior_estimator.state_dict()
                    )
                    torch.save(self.best_model_state_dict, "best_model.pt")
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1
                if self.progress_info:
                    iterator.set_postfix(
                        {
                            "Forecast": forecast_loss.item(),
                            "Reg.": regularisation_loss.item(),
                            "total": loss.item(),
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
        if mpi_rank == 0 and self.log_tensorboard:
            self.writer.flush()
            self.writer.close()
