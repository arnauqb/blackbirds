import torch
import numpy as np
from tqdm import tqdm

from blackbirds.simulate import simulate_and_observe_model


class SMD:
    def __init__(
        self, model, loss_fn, optimizer, gradient_horizon=None, progress_bar=False
    ):
        """
        Simulated Minimum Distance. Finds the point in parameter space that
        minimizes the distance between the model's output and the observed
        data given the loss function `loss_fn`.

        **Arguments:**

        - `model`: A model that inherits from `blackbirds.models.Model`.
        - `loss_fn`: A loss function taking (x,y) arguments, where y is the data and x the simulated value.
        - `optimizer`: A PyTorch optimizer (eg Adam)
        - `gradient_horizon`: The number of steps to look ahead when computing the gradient. If None, defaults to the number of parameters.
        - `progress_bar`: Whether to display a progress bar.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.gradient_horizon = gradient_horizon
        self.progress_bar = progress_bar
        self.loss = []

    def run(
        self,
        data,
        n_epochs=1000,
        max_epochs_without_improvement=100,
        parameters_save_dir="best_parameters.pt",
    ):
        """
        Runs the SMD algorithm for `n_epochs` epochs.

        **Arguments:**

        - `data`: The observed data.
        - `n_epochs`: The number of epochs to run.
        - `max_epochs_without_improvement`: The number of epochs to run without improvement before stopping.
        - `parameters_save_dir`: The directory to save the best parameters to.
        """
        best_loss = np.inf
        epochs_without_improvement = 0
        if self.progress_bar:
            iterator = tqdm(range(n_epochs))
        else:
            iterator = range(n_epochs)
        for _ in iterator:
            self.optimizer.zero_grad()
            parameters = self.optimizer.param_groups[0]["params"][0]
            simulated = simulate_and_observe_model(
                self.model, parameters, self.gradient_horizon
            )
            loss = torch.tensor(0.0)
            for sim, d in zip(simulated, data):
                loss += self.loss_fn(sim, d)
            loss.backward()
            if loss < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
                torch.save(parameters, parameters_save_dir)
            else:
                epochs_without_improvement += 1
            if self.progress_bar:
                iterator.set_postfix(
                    {
                        "loss": loss.item(),
                        "best loss": best_loss,
                        "epochs since improv.": epochs_without_improvement,
                    }
                )

            if epochs_without_improvement >= max_epochs_without_improvement:
                break
            self.optimizer.step()
            self.loss.append(loss.item())
