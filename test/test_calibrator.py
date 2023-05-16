import torch
import numpy as np

from birds.models.random_walk import RandomWalk
from birds.calibrator import Calibrator


class TestCalibrator:
    def test_random_walk(self, TrainableGaussian):
        """
        Tests inference in a random walk model.
        """
        rw = RandomWalk(100)
        true_ps = [0.25, 0.5, 0.75]
        prior = torch.distributions.Normal(0.0, 1.0)
        for diff_mode in ("reverse", "forward"):
            for true_p in true_ps:
                data = rw(torch.tensor([true_p]))
                posterior_estimator = TrainableGaussian(0.5, 0.1)
                optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=1e-2)
                calib = Calibrator(
                    model=rw,
                    posterior_estimator=posterior_estimator,
                    prior=prior,
                    data=data,
                    optimizer=optimizer,
                    diff_mode=diff_mode,
                    w=100.0,
                    progress_bar=False,
                )
                _, best_model_state_dict = calib.run(
                    100, max_epochs_without_improvement=100
                )
                posterior_estimator.load_state_dict(best_model_state_dict)
                # check correct result is within 2 sigma
                assert (
                    np.abs(posterior_estimator.mu.item() - true_p)
                    < 2 * posterior_estimator.sigma.item()
                )
