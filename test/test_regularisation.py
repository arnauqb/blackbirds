import torch
import numpy as np
from birds.regularisation import compute_regularisation_loss

class TestRegularisation:
    def test_regularisation(self):
        n_samples = 100000
        # define two normal distributions
        dist1 = torch.distributions.Normal(0, 1)
        dist2 = torch.distributions.Normal(0, 1)
        # check that the KL divergence is 0
        assert np.isclose(compute_regularisation_loss(dist1, dist2, n_samples), 0.)
        # define two normal distributions with different means
        dist1 = torch.distributions.Normal(0, 1)
        dist2 = torch.distributions.Normal(1, 1)
        # check that the KL divergence is the right result
        assert np.isclose(compute_regularisation_loss(dist1, dist2, n_samples), 0.5, rtol=1e-2)



