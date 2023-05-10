import torch
import numpy as np
from birds.models.random_walk import RandomWalk

class TestRandomWalk:
    def test__result(self):
        rw = RandomWalk(1000, 0.3)
        assert rw.n_timesteps == 1000
        assert rw.p == 0.3
        result = rw()
        avg_steps_forward = 0.3 * 1000
        avg_steps_backward = 0.7 * 1000
        assert np.isclose(result[-1], avg_steps_forward - avg_steps_backward, rtol=1e-1)

    def test__gradient(self):
        p = torch.tensor(0.4, requires_grad=True)
        rw = RandomWalk(10, p)
        assert rw.n_timesteps == 10
        assert rw.p == 0.4
        assert rw.tau_softmax == 0.1
        result = rw()
        assert result.shape == (10,)
        result[-1].backward()
        assert p.grad is not None


