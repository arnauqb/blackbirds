from blackbirds.forecast import compute_and_differentiate_forecast_loss
from blackbirds.models.june import June
from blackbirds.posterior_estimators import TrainableGaussian
from memory_profiler import profile
import torch
import numpy as np
import yaml

import sys
sys.path.append("./experiments/gradient_horizon")
from june import make_model, make_flow4, make_prior, true_parameters, _all_parameters, _all_no_seed_parameters

device = "cuda:1"
n_parameters = len(true_parameters) 
means = torch.tensor([-3.5] + list(-1.0 * np.ones(len(_all_parameters)-1)), dtype=torch.float, device=device)
#means = torch.tensor(list(-1.0 * np.ones(len(_all_parameters)-1)), dtype=torch.float, device=device)
true_parameters = means.clone()
#flow = make_flow4(n_parameters, device)
flow = TrainableGaussian(means, 0.01, device=device) #make_flow4(n_parameters, device)
prior = make_prior(n_parameters, device)
loss_fn = torch.nn.MSELoss()
#flow.load_state_dict(torch.load("../best_model.pt", map_location="cuda:0"))


def run():
    config_file = "./examples/june_config.yaml"
    config = yaml.safe_load(open(config_file))
    days = 200
    print("Initial allocation")
    print(torch.cuda.memory_allocated(device)/ 1e6)
    config["timer"]["total_days"] = days
    #config["data_path"] = "../gradabm-june/worlds/camden_leisure_1.pkl"
    config["data_path"] = "/cosma7/data/dp004/dc-quer1/gradabm_june_graphs/camden_leisure_1.pkl"
    config["system"]["device"] = device
    model = June(config, _all_parameters)
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
