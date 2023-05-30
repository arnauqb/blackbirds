from birds.forecast import compute_and_differentiate_forecast_loss
from birds.models.june import June
from birds.posterior_estimators import TrainableGaussian
import torch
import numpy as np
import yaml

import sys
sys.path.append("./experiments/gradient_horizon")
from june import make_model, make_prior, true_parameters, _all_parameters

device = "cpu"
n_parameters = len(true_parameters) 
flow = TrainableGaussian(torch.zeros(len(true_parameters)), 0.1) #make_flow4(n_parameters, device)
prior = make_prior(n_parameters, device)
config_file = "./examples/june_config.yaml"
config = yaml.safe_load(open(config_file))
config["data_path"] = "../gradabm-june/worlds/camden_leisure_1.pkl"
model = June(config, _all_parameters)
loss_fn = torch.nn.MSELoss()
#flow.load_state_dict(torch.load("../best_model.pt", map_location="cuda:0"))
true_data = model.run_and_observe(true_parameters)

def run():
    compute_and_differentiate_forecast_loss(loss_fn=loss_fn,
                                        model=model,
                                        posterior_estimator=flow,
                                        n_samples = 5,
                                        observed_outputs=true_data,
                                        diff_mode="forward",
                                       )

