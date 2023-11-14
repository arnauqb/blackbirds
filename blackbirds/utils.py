import torch
import normflows as nf


def soft_maximum(a: torch.Tensor, b: torch.Tensor, k: float):
    """
    Soft differentiable maximum function.

    **Arguments:**

    - `a`: First input tensor.
    - `b`: Second input tensor.
    - `k`: Hardness.
    """
    return torch.log(torch.exp(k * a) + torch.exp(k * b)) / k


def soft_minimum(a: torch.Tensor, b: torch.Tensor, k: float):
    """
    Soft differentiable minimum function.

    **Arguments:**

    - `a`: First input tensor.
    - `b`: Second input tensor.
    - `k`: Hardness.
    """
    return -soft_maximum(-a, -b, k)


class Sigmoid(nf.flows.Flow):
    def __init__(self, min_values, max_values):
        super().__init__()
        self.min_values = min_values
        self.max_values = max_values

    def inverse(self, z):
        logz = torch.log(z - self.min_values)
        log1mz = torch.log(self.max_values - z)
        z = logz - log1mz
        sum_dims = list(range(1, z.dim()))
        log_det = -torch.sum(logz, dim=sum_dims) - torch.sum(log1mz, dim=sum_dims)
        return z, log_det

    def forward(self, z):
        sum_dims = list(range(1, z.dim()))
        ls = torch.sum(torch.nn.functional.logsigmoid(z), dim=sum_dims)
        mls = torch.sum(torch.nn.functional.logsigmoid(-z), dim=sum_dims)
        lls = torch.sum(torch.log(self.max_values - self.min_values))
        log_det = ls + mls + lls
        z = self.min_values + (self.max_values - self.min_values) * torch.sigmoid(z)
        return z, log_det
