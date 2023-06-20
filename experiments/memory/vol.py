from birds.forecast import compute_and_differentiate_forecast_loss
from birds.models.rama_cont import RamaCont
from birds.posterior_estimators import TrainableGaussian
from memory_profiler import profile
import torch
import numpy as np
import yaml

    
class LogModel(RamaCont):
    def initialize(self, params):
        return super().initialize(10 ** params)
    def step(self, params, x):
        return super().step(10 ** params, x)

device = "cpu"
true_parameters = torch.log10(torch.tensor([0.1, 1, 1e-3, 10]))
n_parameters = len(true_parameters) 

#flow = make_flow4(n_parameters, device)
flow = TrainableGaussian(true_parameters.cpu().numpy(), 0.01, device=device) #make_flow4(n_parameters, device)
prior = torch.distributions.MultivariateNormal(torch.zeros(4), 1.0 * torch.eye(len(true_parameters)))

loss_fn = torch.nn.MSELoss()

def run():
    print("Initial allocation")
    print(torch.cuda.memory_allocated(device)/ 1e6)
    model = LogModel(n_agents = 100_000, n_timesteps=1000, s=0.1, sigmoid_k=5.0)
    true_data = model.run_and_observe(true_parameters)
    compute_and_differentiate_forecast_loss(loss_fn=loss_fn,
                                            model=model,
                                            posterior_estimator=flow,
                                            n_samples = 1,
                                            observed_outputs=true_data,
                                            diff_mode="forward",
                                            device=device
                                           )

run()
