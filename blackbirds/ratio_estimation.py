import logging
import torch
import torch.nn as nn
import torch.optim
from tqdm import trange

logger = logging.getLogger("ratio")

def train(
    train_x0s: torch.Tensor,
    train_x1s: torch.Tensor,
    val_x0s: torch.Tensor,
    val_x1s: torch.Tensor,
    ratio_estimator: nn.Module,
    ratio_optimiser: torch.optim.Optimizer,
    batch_size: int = 50,
    max_num_epochs: int = 200,
    best_loss: float = float('inf'),
    max_iterations_without_val_loss_improvement: int = 20,
    outloc: str = "./best_re.pth",
    stop_at_loss: float = 0.
    ):

    """
    Training script for binary ratio estimators

    **Arguments**

    - `train_x0s`: Training examples for the negative class.
    - `train_x1s`: Training examples for the positive class.
    - `val_x0s`: Validation examples for the negative class.
    - `val_x1s`: Validation examples for the positive class.
    - `ratio_estimator`: torch.nn.Module mapping from input space to logit.
    - `ratio_optimiser`: torch.optim.Optimizer specifying how to optimise ratio_estimator.
    - `batch_size`: Integer denoting the batch size used for training.
    - `max_num_epochs`: An integer denoting the maximum number of epochs to train for.
    - `max_iterations_without_val_loss_improvement`: Integer specifying the maximum number 
       of epochs to train for without any improvement in the validation loss. (This is a
       form of early stopping.)
    - `outloc`: A string specifying the location at which the best ratio estimator is to be
       saved for later use.
    - `stop_at_loss`: A float specifying the loss below which training should be stopped.
       Useful when a lower bound on the loss is known.
    """

    iterator = trange(max_num_epochs)
    loss_hist = []
    N_TRAIN = train_x0s.shape[0]
    m = 0

    splus_min = nn.Softplus(beta=-1.)

    for epoch in iterator:
        idx = 0
        while idx < N_TRAIN:
            ratio_optimiser.zero_grad()
            end_idx = idx + batch_size
            x0s_, x1s_ = train_x0s[idx:end_idx], train_x1s[idx:end_idx]
            x0_out = ratio_estimator.forward(x0s_)
            loss = -splus_min(x0_out).sum()
            x1_out = ratio_estimator.forward(x1s_)
            loss += (x1_out - splus_min(x1_out)).sum()
            loss /= (2 * x0s_.shape[0])
            loss.backward()
            ratio_optimiser.step()
            idx += batch_size
        with torch.no_grad():
            # Compute val_joint_loss and product
            x0_out = ratio_estimator.forward(val_x0s)
            val_loss = -splus_min(x0_out).sum()
            x1_out = ratio_estimator.forward(val_x1s)
            val_loss += (x1_out - splus_min(x1_out)).sum()
            val_loss /= (2 * x0_out.shape[0])
            if val_loss < stop_at_loss:
                logger.info("Training stopped – best loss surpassed.")
                return ratio_estimator, loss_hist
            loss_hist.append(val_loss.item())
            m += 1
            if val_loss.item() < best_loss:
                if not outloc is None:
                    torch.save(ratio_estimator.state_dict(), outloc)
                best_loss = val_loss.item()
                m = 0
            iterator.set_postfix({"best_val_loss": best_loss, "current_val_loss":val_loss.item()})
            if m >= max_iterations_without_val_loss_improvement:
                logger.info("Training stopped - converged.")
                return ratio_estimator, loss_hist
    logger.ingo("Training stopped - max number of iterations reached.")
    return ratio_estimator, loss_hist
