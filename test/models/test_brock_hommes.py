import torch

from birds.models.brock_hommes import BrockHommes


class TestBrockHomes:
    def test__gradient_propagates(self):
        model = BrockHommes()
        params = torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4, 0.1, 0.01], requires_grad=True
        )
        x = model(params)[0]
        x.sum().backward()
        assert params.grad is not None
