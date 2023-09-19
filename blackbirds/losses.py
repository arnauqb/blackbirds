from typing import Union
import torch

from blackbirds.models.model import Model
from blackbirds.simulate import simulate_and_observe_model


class SingleOutput_SimulateAndMSELoss:

    """
    Computes MSE between observed data y and simulated data at theta (to be passed during __call__).

    **Arguments:**

    - `model`: An instance of a Model. The model that you'd like to "fit".
    - `gradient_horizon`: Specifies the gradient horizon to use. None implies infinite horizon.
    """

    def __init__(self, model: Model, gradient_horizon: Union[int, None] = None):
        self.loss = torch.nn.MSELoss()
        self.model = model
        self.gradient_horizon = gradient_horizon

    def __call__(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ):
        x = simulate_and_observe_model(self.model, theta, self.gradient_horizon)[0]
        return self.loss(x, y)


class UnivariateMMDLoss:
    def __init__(self, y: torch.Tensor):
        """
        Computes MMD between data y and simulated output x (to be passed during call).

        **Arguments:**

        - `y`: torch.Tensor containing a single univariate time series.
        """
        assert isinstance(y, torch.Tensor), "y is assumed to be a torch.Tensor here"
        try:
            assert (
                len(y.shape) == 1
            ), "This class assumes y is a single univariate time series"
        except AssertionError:
            assert (
                len(y.shape) == 2
            ), "If not a 1D Tensor, y must be at most 2D of shape (1, T)"
            assert (
                y.shape[1] == 1
            ), "This class assumes y is a single univariate time series. This appears to be a batch of data."
            y = y.reshape(-1)
        self.device = y.device
        self.y = y
        self.y_matrix = self.y.reshape(1, -1, 1)
        yy = torch.cdist(self.y_matrix, self.y_matrix)
        yy_sqrd = torch.pow(yy, 2)
        self.y_sigma = torch.median(yy_sqrd)
        ny = self.y.shape[0]
        self.kyy = (
            torch.exp(-yy_sqrd / self.y_sigma) - torch.eye(ny, device=self.device)
        ).sum() / (ny * (ny - 1))

    def __call__(
        self,
        x: torch.Tensor,
    ):
        assert isinstance(x, torch.Tensor), "x is assumed to be a torch.Tensor here"
        try:
            assert (
                len(x.shape) == 1
            ), "This class assumes x is a single univariate time series"
        except AssertionError:
            assert (
                len(x.shape) == 2
            ), "If not a 1D Tensor, x must be at most 2D of shape (1, T)"
            assert (
                x.shape[1] == 1
            ), "This class assumes x is a single univariate time series. This appears to be a batch of data."
            x = x.reshape(-1)

        nx = x.shape[0]
        x_matrix = x.reshape(1, -1, 1)
        kxx = torch.exp(-torch.pow(torch.cdist(x_matrix, x_matrix), 2) / self.y_sigma)
        kxx = (kxx - torch.eye(nx, device=self.device)).sum() / (nx * (nx - 1))
        kxy = torch.exp(
            -torch.pow(torch.cdist(x_matrix, self.y_matrix), 2) / self.y_sigma
        )
        kxy = kxy.mean()
        return kxx + self.kyy - 2 * kxy


class SingleOutput_SimulateAndMMD:
    """
    Example implementation of a loss that simulates from the model and computes the MMD
    between the model output and observed data y. (This treats the entries in y and in
    the simulator output as exchangeable.)

    **Arguments:**

    - `y`: torch.Tensor containing a single univariate time series.
    - `model`: An instance of a Model.
    - `gradient_horizon`: An integer or None. Sets horizon over which gradients are retained. If None, infinite horizon used.
    """
    def __init__(
        self, y: torch.Tensor, model: Model, gradient_horizon: Union[int, None] = None
    ):
        self.mmd_loss = UnivariateMMDLoss(y)
        self.model = model
        self.gradient_horizon = gradient_horizon

    def __call__(self, theta: torch.Tensor, y: torch.Tensor):
        x = simulate_and_observe_model(self.model, theta, self.gradient_horizon)[0]
        return self.mmd_loss(x)
