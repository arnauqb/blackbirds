from birds.forecast import compute_and_differentiate_forecast_loss
from birds.models.rama_cont import RamaCont
from birds.posterior_estimators import TrainableGaussian
#from memory_profiler import profile
import torch
import numpy as np
import yaml


class LogModel(RamaCont):
    def initialize(self, params):
        return super().initialize(10**params)

    def step(self, params, x):
        return super().step(10**params, x)


device = "cpu"
true_parameters = torch.tensor([0.1, 1, 1e-1, 10])
n_parameters = len(true_parameters)

# flow = make_flow4(n_parameters, device)
flow = TrainableGaussian(
    true_parameters.cpu().numpy(), 0.001, device=device
)  # make_flow4(n_parameters, device)
prior = torch.distributions.MultivariateNormal(
    torch.zeros(4), 1.0 * torch.eye(len(true_parameters))
)

loss_fn = torch.nn.MSELoss()
model = RamaCont(n_agents=10000000, n_timesteps=100, s=0.1, sigmoid_k=5.0)
true_data = model.run_and_observe(true_parameters)

@profile
def run1():
    compute_and_differentiate_forecast_loss(
        loss_fn=loss_fn,
        model=model,
        posterior_estimator=flow,
        n_samples=1,
        observed_outputs=true_data,
        diff_mode="reverse",
        device=device,
        jacobian_chunk_size=1,
    )

@profile
def run2():
    compute_and_differentiate_forecast_loss(
        loss_fn=loss_fn,
        model=model,
        posterior_estimator=flow,
        n_samples=1,
        observed_outputs=true_data,
        diff_mode="forward",
        device=device,
        jacobian_chunk_size=1,
    )


run1()
run2()
