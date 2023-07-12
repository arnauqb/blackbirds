import torch
import numpy as np
from tqdm import tqdm


class SMD:
    def __init__(self, loss, optimizer, gradient_horizon=None, progress_bar=False):
        """
        Simulated Minimum Distance. Finds the point in parameter space that
        minimizes the distance between the model's output and the observed
        data given the loss function `loss_fn`.

        **Arguments:**

        - `loss` : A callable that returns a (differentiable) loss. Needs to take (parameters, data) as input and return a scalar tensor.
        - `optimizer`: A PyTorch optimizer (eg Adam)
        - `gradient_horizon`: The number of steps to look ahead when computing the gradient. If None, defaults to the number of parameters.
        - `progress_bar`: Whether to display a progress bar.
        """
        self.loss_fn = loss
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
            loss = self.loss_fn(parameters, data)
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
