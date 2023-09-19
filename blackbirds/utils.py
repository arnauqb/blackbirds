import torch


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
