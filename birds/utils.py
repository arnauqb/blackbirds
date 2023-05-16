import torch


def soft_maximum(a, b, k):
    """
    Soft differentiable maximum function.

    Arguments:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.
        k (float): Hardness.
    """
    return torch.log(torch.exp(k * a) + torch.exp(k * b)) / k


def soft_minimum(a, b, k):
    """
    Soft differentiable minimum function.

    Arguments:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.
        k (float): Hardness.
    """
    return -soft_maximum(-a, -b, k)
