from birds.jacfwd import jacfwd
from torch.func import jacrev
import torch


class TestJacFwd:
    def test__vs_pytorch(self):
        func = lambda x: x**2 + 3 * x
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert torch.allclose(jacfwd(func)(x), torch.func.jacfwd(func)(x))
        func = lambda x, y: x**2 + y
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        assert torch.allclose(jacfwd(func)(x, y), torch.func.jacfwd(func)(x, y))
