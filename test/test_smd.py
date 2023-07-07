import torch
import numpy as np

from blackbirds.smd import SMD
from blackbirds.models.random_walk import RandomWalk


class TestSMD:
    """
    Tests the methods of Simulated Minimum Distance (SMD).
    """

    def test__rw(self):
        """
        Tests the method with the RandomWalk model.
        """
        rw = RandomWalk(n_timesteps=100)
        true_p = torch.logit(torch.tensor([0.25]))
        true_data = rw.run_and_observe(true_p)
        parameters = torch.logit(torch.tensor([0.5]))
        parameters.requires_grad = True
        loss_fn = torch.nn.MSELoss()
        optim = torch.optim.Adam([parameters], lr=1e-2)
        smd = SMD(model=rw, optimizer=optim, loss_fn=loss_fn)
        smd.run(true_data, n_epochs=1000)
        assert np.min(smd.loss) < 5
        best_parameters = torch.sigmoid(torch.load("best_parameters.pt"))
        assert torch.isclose(best_parameters, torch.tensor(0.25), rtol=1e-2, atol=1e-2)
