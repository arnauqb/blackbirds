from blackbirds.model import Model
from blackbirds.simulate import simulate_and_observe

import torch

class SingleOutput_SmulateAndMSELoss:

    """
    Computes MSE between observed data y and simulated data at theta (to be passed during __call__).

    **Arguments**

    - `model`: An instance of a Model. The model that you'd like to "fit".
    - `gradient_horizon`: Specifies the gradient horizon to use. None implies infinite horizon.
    """

    def __init__(
        self, 
        model: Model, 
        gradient_horizon: int | None = None
    ):

        self.loss = torch.nn.MSELoss()
        self.model = model
        self.gradient_horizon = gradient_horizon

    def __call__(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ):

        x = simulate_and_observe_model(
            self.model, 
            theta, 
            self.gradient_horizon
        )[0]
        return self.loss(x, y)

class UnivariateMMDLoss:
    def __init__(
        self, 
        y: torch.Tensor
    ):

        """
        Computes MMD between data y and simulated output x (to be passed during call).

        Assumes y is a torch.Tensor consisting of a single univariate time series.
        """

        assert isinstance(y, torch.Tensor), "y is assumed to be a torch.Tensor here"
        assert len(y.shape) == 1, "This class assumes y is a single univariate time series"

        self.y = y
        self.y_matrix = self.y.reshape(1,-1,1)
        yy = torch.cdist(self.y_matrix, self.y_matrix)
        yy_sqrd = torch.pow(yy, 2)
        self.y_sigma = torch.median(yy_sqrd)
        ny = self.y.shape[0]
        self.kyy = (
            torch.exp( 
                -yy_sqrd / self.y_sigma 
            ) 
            - torch.eye(ny)
        ).sum() / (ny * (ny - 1))
        
    def __call__(
        self, 
        x: torch.Tensor,
    ):

        assert isinstance(x, torch.Tensor), "x is assumed to be a torch.Tensor here"
        assert len(x.shape) == 1, "This class assumes x is a single univariate time series"

        nx = x.shape[0]
        x_matrix = x.reshape(1,-1,1)
        kxx = torch.exp( 
            -torch.pow(
                torch.cdist(x_matrix, x_matrix), 
             2) 
            / self.y_sigma 
        )
        kxx = (kxx - torch.eye(nx)).sum() / (nx * (nx - 1))
        kxy = torch.exp( - torch.pow(torch.cdist(x_matrix, self.y_matrix), 2) / self.y_sigma )
        kxy = kxy.mean()
        return kxx + self.kyy - 2 * kxy
    
class SingleOutput_SimulateAndMMD:

    """
    Example implementation of a loss that simulates from the model and computes the MMD
    between the model output and observed data y. (This treats the entries in y and in
    the simulator output as exchangeable.)

    **Arguments**

    - `y`: torch.Tensor containing a single univariate time series.
    - `model`: An instance of a Model.
    - `gradient_horizon`: An integer or None. Sets horizon over which gradients are retained. If None, infinite horizon used.
    """

    def __init__(
        self, 
        y: torch.Tensor, 
        model: Model, 
        gradient_horizon: int | None = None
    ):

        self.mmd_loss = MMDLoss(y)
        self.model = model
        self.gradient_horizon = gradient_horizon
        
    def __call__(
        self, 
        theta: torch.Tensor,
        y: torch.Tensor
    ):

        x = simulate_and_observe_model(self.model, theta, self.gradient_horizon)[0]
        return self.mmd_loss(x)
