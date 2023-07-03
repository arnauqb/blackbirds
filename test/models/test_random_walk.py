import torch
import numpy as np
from blackbirds.models.random_walk import RandomWalk


class TestRandomWalk:
    def test__result(self):
        n_timesteps = 1000
        rw = RandomWalk(n_timesteps)
        assert rw.n_timesteps == n_timesteps
        logit_p = torch.logit(torch.tensor(0.3))
        x = rw.run(logit_p)
        assert x.shape == (n_timesteps + 1, 1)
        trajectory = rw.observe(x)[0]
        avg_steps_forward = 0.3 * n_timesteps
        avg_steps_backward = 0.7 * n_timesteps
        assert np.isclose(
            trajectory[-1], avg_steps_forward - avg_steps_backward, rtol=1e-1
        )

    def test__gradient(self):
        p = torch.tensor(0.4, requires_grad=True)
        n_timesteps = 10
        rw = RandomWalk(n_timesteps)
        assert rw.n_timesteps == n_timesteps
        assert rw.tau_softmax == 0.1
        x = rw.observe(rw.run(p))[0]
        assert x.shape == (n_timesteps + 1, 1)
        x[-1].backward()
        assert p.grad is not None
