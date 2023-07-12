import pytest
import torch
import random
import numpy as np
import normflows as nf

from blackbirds.models import Model
from blackbirds.simulate import simulate_and_observe_model


@pytest.fixture(autouse=True)
def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


class MockPosteriorEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.tensor(1.0, requires_grad=True)

    def sample(self, n):
        x = 2 * self.p * torch.ones((n, 2))
        return x, self.log_prob(x)

    def log_prob(self, x):
        if len(x.shape) == 1:
            return torch.zeros(1) * self.p
        else:
            return torch.zeros(x.shape[0]) * self.p

    def forward(self, n):
        return self.sample(n)


class MockModel(Model):
    def __init__(self):
        super().__init__()
        self.n_timesteps = 2

    def trim_time_series(self, x):
        return x[-1:]

    def initialize(self, params):
        x = (params**2).reshape(1, -1)
        return x

    def step(self, params, x):
        return (params**2).reshape(1, -1)

    def observe(self, x):
        return [x]


class MockLoss:
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, parameters, data):
        model_outputs = simulate_and_observe_model(model=self.model, params=parameters)
        return self.loss_fn(model_outputs[0], data[0])


@pytest.fixture(name="mock_estimator")
def make_mock_posterior():
    return MockPosteriorEstimator()


@pytest.fixture(name="mock_model")
def make_mock_model():
    return MockModel()


@pytest.fixture(name="mock_loss")
def make_mock_loss(mock_model):
    return MockLoss(mock_model)


@pytest.fixture(name="make_flow")
def make_flow_fn():
    def make_flow(n_parameters):
        K = 4
        torch.manual_seed(0)
        flows = []
        for i in range(K):
            flows.append(
                nf.flows.MaskedAffineAutoregressive(n_parameters, 20, num_blocks=2)
            )
        q0 = nf.distributions.DiagGaussian(n_parameters)
        nfm = nf.NormalizingFlow(q0=q0, flows=flows)
        return nfm

    return make_flow
