import torch

from birds.models.volatality import ClusteredVolatality

class TestClusteredVolatality:
    def test_gradient_propagates(self):
        model = ClusteredVolatality()

        params = torch.tensor([0.01, 10], requires_grad=True) # [q, lam]
        output = model.run(params)[-1]

        output.sum().backward()
        assert params.grad is not None