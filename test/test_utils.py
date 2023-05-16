import torch

from birds.utils import soft_minimum, soft_maximum


class TestSoftMaximum:
    def test__soft_maximum(self):
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        k = 3.0
        maxv = soft_maximum(a, b, k)
        assert torch.isclose(maxv, b, rtol=1e-2)


class TestSoftMinimum:
    def test__soft_minimum(self):
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        k = 3
        maxv = soft_minimum(a, b, k)
        assert torch.isclose(maxv, a, rtol=2e-2)
