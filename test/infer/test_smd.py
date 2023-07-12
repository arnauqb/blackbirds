import torch
import numpy as np

from blackbirds.infer import SMD
from blackbirds.simulate import simulate_and_observe_model
from blackbirds.models.random_walk import RandomWalk

class L2Loss:
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
    def __call__(self, params, data):
        simulated = simulate_and_observe_model(self.model, params)
        return self.loss_fn(simulated[0], data[0])

class TestSMD:
    """
    Tests the methods of Simulated Minimum Distance (SMD).
    """

    def test__rw(self):
        """
        Tests the method with the RandomWalk model.
        """
        rw = RandomWalk(n_timesteps=100)
        loss = L2Loss(rw)
        true_p = torch.logit(torch.tensor([0.25]))
        true_data = rw.run_and_observe(true_p)
        parameters = torch.logit(torch.tensor([0.5]))
        parameters.requires_grad = True
        optim = torch.optim.Adam([parameters], lr=1e-2)
        smd = SMD(loss=loss, optimizer=optim)
        smd.run(true_data, n_epochs=200)
        assert np.min(smd.loss) < 20
        best_parameters = torch.sigmoid(torch.load("best_parameters.pt"))
        assert torch.isclose(best_parameters, torch.tensor(0.25), rtol=5e-2, atol=5e-2)
