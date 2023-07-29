import copy
import logging
import torch
import torch.nn as nn
import torch.optim
from tqdm import trange
from typing import Callable

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("ratio")


class RatioEstimator(nn.Module):

    """
    A generic implementation of a density ratio estimator.

    **Arguments**

    - `classifier_network`: A torch.nn.Module for the binary classifier to use. Must
       have a .forward method mapping from an input vector to logit.
    - `summary_network`: A torch.nn.Module specifying how to cast simulator output
       to a low-dimensional vector. Default: nn.Identity() (i.e. no dimensionality
       reduction).
    """

    def __init__(
        self, 
        classifier_network: nn.Module,
        summary_network: nn.Module = nn.Identity()
    ):

        super().__init__()

        self._sn = summary_network
        self._cn = classifier_network
        self._spn = nn.Softplus(beta=-1.)

    def sn_forward(
        self, 
        x: torch.Tensor
    ):

        return self._sn(x)

    def log_ratio(
        self,
        sx: torch.Tensor,
        theta: torch.Tensor
    ):

        if not len(theta.shape) == 2:
            theta = theta.unsqueeze(0)
        data = torch.cat((theta, sx), dim=-1)
        data = self._cn(data).reshape(-1)
        return data

    def forward(
        self, 
        sx: torch.Tensor, 
        theta: torch.Tensor, 
        prior=False
    ):

        data = self.log_ratio(sx, theta)
        this_loss = -self._spn(data)
        if prior:
            this_loss = this_loss + data
        this_loss = this_loss.mean()
        return this_loss


def train(
    train_x: torch.Tensor,
    train_theta: torch.Tensor,
    val_x: torch.Tensor,
    val_theta: torch.Tensor,
    ratio_estimator: nn.Module,
    ratio_optimiser: torch.optim.Optimizer,
    ratio_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    batch_size: int = 50,
    max_num_epochs: int = 200,
    best_loss: float = float('inf'),
    max_iterations_without_val_loss_improvement: int = 20,
    outloc: str | None = None,
    stop_at_loss: float = 0.,
    tensorboard_log_dir: str | None = None,
    gradient_clipping_norm: float | None = None
    ):

    """
    Training script for binary ratio estimator given by RatioEstimator class.

    **Arguments**

    - `train_x`: The simulator outputs used in training.
    - `train_theta`: The parameters used in training.
    - `val_x`: The simulator output reserved for validation.
    - `val_theta`: The parameters reserved for validation.
    - `ratio_estimator`: torch.nn.Module mapping from input space to logit.
    - `ratio_optimiser`: torch.optim.Optimizer specifying how to optimise ratio_estimator.
    - `ratio_scheduler`: torch.optim.lr_scheduler.LRScheduler or None. Specifies learning
       rate scheduler to use, if any. Default None.
    - `batch_size`: Integer denoting the batch size used for training.
    - `max_num_epochs`: An integer denoting the maximum number of epochs to train for.
    - `max_iterations_without_val_loss_improvement`: Integer specifying the maximum number 
       of epochs to train for without any improvement in the validation loss. (This is a
       form of early stopping.)
    - `outloc`: A string specifying the location at which the best ratio estimator is to be
       saved for later use.
    - `stop_at_loss`: A float specifying the loss below which training should be stopped.
       Useful when a lower bound on the loss is known.
    - `tensorboard_log_dir`: A string specifying the location at which to store tensorboard
    """

    iterator = trange(max_num_epochs, position=0, leave=True)
    loss_hist = []
    N_TRAIN = train_x.shape[0]
    N_VAL = val_x.shape[0]
    m = 0
    best_ratio_estimator = copy.deepcopy(ratio_estimator)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    for epoch in iterator:
        idx = 0
        epoch_train_loss = 0.
        epoch_val_loss = 0.
        while idx < N_TRAIN:

            ratio_optimiser.zero_grad()
            end_idx = idx + batch_size

            joint_x, joint_theta = train_x[idx:end_idx], train_theta[idx:end_idx]
            joint_sx = ratio_estimator.sn_forward(joint_x)
            joint_loss = ratio_estimator.forward(joint_sx, joint_theta, prior=False)

            THIS_N = joint_x.shape[0] # In case we're at the last batch and there are fewer than batch_size elements

            product_loss = 0.
            for i in range(1, THIS_N):
                rolled_thetas = torch.roll(joint_theta, i, 0)
                product_loss += ratio_estimator.forward(joint_sx, rolled_thetas, prior=True)
            loss = product_loss / (THIS_N - 1) + joint_loss

            #rolled_thetas = torch.roll(joint_theta, torch.randint(low=1, high=THIS_N, size=(1,)).item(), 0)
            #product_loss = ratio_estimator.forward(joint_sx, rolled_thetas, prior=True)
            #loss = product_loss + joint_loss

            epoch_train_loss += loss.item()
            logger.info("Loss = {0}".format(loss.item()))

            loss.backward()
            if gradient_clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(ratio_estimator.parameters(), gradient_clipping_norm)
            #print([p.grad for p in ratio_estimator.parameters()])
            ratio_optimiser.step()
            idx += batch_size
        with torch.no_grad():
            # Compute val_joint_loss and product
            val_sx = ratio_estimator.sn_forward(val_x)
            val_joint_loss = ratio_estimator.forward(val_sx, val_theta, prior=False)
            val_product_loss = 0.

            for i in range(1, N_VAL):
                rolled_thetas = torch.roll(val_theta, i, 0)
                val_product_loss += ratio_estimator.forward(val_sx, rolled_thetas, prior=True)
            val_loss = val_product_loss / (N_VAL - 1) + val_joint_loss
            epoch_val_loss += val_loss.item()

            #rolled_thetas = torch.roll(val_theta, torch.randint(low=1, high=N_VAL, size=(1,)).item(), 0)
            #val_product_loss = ratio_estimator.forward(val_sx, rolled_thetas, prior=True)
            #val_loss = val_product_loss + val_joint_loss

            loss_hist.append(val_loss.item())
            m += 1
            if val_loss.item() < best_loss:
                best_ratio_estimator = copy.deepcopy(ratio_estimator)
                if not outloc is None:
                    torch.save(best_ratio_estimator.state_dict(), outloc)
                best_loss = val_loss.item()
                m = 0
            if val_loss < stop_at_loss:
                logger.info("Training stopped – best loss surpassed.")
                return best_ratio_estimator, loss_hist
            iterator.set_postfix({"train_loss": loss.item(), 
                                  "best_val_loss": best_loss, 
                                  "current_val_loss":val_loss.item(),
                                  "steps since last improvement":m})
            if m >= max_iterations_without_val_loss_improvement:
                logger.info("Training stopped - converged.")
                return best_ratio_estimator, loss_hist
            if not ratio_scheduler is None:
                ratio_scheduler.step(val_loss)
        writer.add_scalar("loss/train", epoch_train_loss, epoch)
        writer.add_scalar("loss/val", epoch_val_loss, epoch)
    logger.info("Training stopped - max number of iterations reached.")
    writer.flush()
    writer.close()
    return best_ratio_estimator, loss_hist
