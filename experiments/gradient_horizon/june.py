"""
This scripts shows how to run BIRDS in parallel using MPI4PY.
The parallelization is done across the number of parameters that are sampled
in each epoch from the posterior candidate.

As an example we consider the SIR model.
"""
import argparse
import torch
import yaml
import normflows as nf
import numpy as np

from birds.models.june import June
from birds.calibrator import Calibrator
from birds.mpi_setup import mpi_rank

_all_no_seed_parameters = [
    "beta_household",
    "beta_company",
    "beta_school",
    "beta_university",
    "beta_pub",
    "beta_grocery",
    "beta_gym",
    "beta_cinema",
    "beta_visit",
    "beta_care_visit",
    "beta_care_home",
]

def loss_fn(x, y):
    mask = (x > 0) & (y > 0)
    x = x[mask].log10()
    y = y[mask].log10()
    return torch.nn.MSELoss()(x, y)


def make_flow(n_parameters, device):
    base = nf.distributions.base.DiagGaussian(n_parameters)
    num_layers = 5
    latent_size = n_parameters
    flows = []
    for _ in range(num_layers):
        param_map = nf.nets.MLP([n_parameters//2 + 1, 50, 50, 2], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(latent_size, mode="swap"))
    flow = nf.NormalizingFlow(base, flows)
    return flow.to(device)

def make_nsf_flow(n_parameters, device):
    K = 3
    torch.manual_seed(0)
    latent_size = n_parameters
    hidden_units = 128
    hidden_layers = 3
    
    flows = []
    for _ in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    
    # Set prior and q0
    q0 = nf.distributions.DiagGaussian(n_parameters, trainable=False)
        
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm.to(device)


def make_prior(n_parameters, device):
    prior = torch.distributions.MultivariateNormal(
        0.5 * torch.ones(n_parameters, device=device),
        0.5 * torch.eye(n_parameters, device=device),
    )
    return prior


def train_flow(model, true_data, n_epochs, n_samples_per_epoch, n_parameters, device):
    # torch.manual_seed(0)
    prior = make_prior(n_parameters, device)
    estimator = make_flow(n_parameters, device)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-3)
    calibrator = Calibrator(
        model=model,
        posterior_estimator=estimator,
        prior=prior,
        data=true_data,
        optimizer=optimizer,
        n_samples_per_epoch=n_samples_per_epoch,
        w=1e-2,
        n_samples_regularisation=10_000,
        forecast_loss=loss_fn,
        log_tensorboard=True,
        gradient_estimation_method="pathwise",
        gradient_horizon=1,
        diff_mode="reverse",
        device=device,
        # gradient_clipping_norm=0.1,
    )

    calibrator.run(n_epochs=100000, max_epochs_without_improvement=np.inf)
    return calibrator


def make_model(config_file, device):
    config = yaml.safe_load(open(config_file))
    config["system"]["device"] = device
    config[
        "data_path"
    ] = "/cosma7/data/dp004/dc-quer1/gradabm_june_graphs/camden_leisure_1.pkl"
    model = June(
        config,
        parameters_to_calibrate=_all_no_seed_parameters #("beta_household", "beta_company", "beta_school"),
    )
    return model


if __name__ == "__main__":
    # parse arguments from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", default=["cpu"], nargs="+")
    args = parser.parse_args()

    # device of this rank
    device = args.device_ids[mpi_rank]
    true_parameters = 0.4 * torch.ones(len(_all_no_seed_parameters))#torch.tensor([0.9, 0.3, 0.6])
    n_parameters = len(true_parameters)
    config_file = "./examples/june_config.yaml"
    model = make_model(config_file, device)
    with torch.no_grad():
        true_data = model.run_and_observe(true_parameters)
    np.savetxt("./true_data.txt", true_data[0].cpu().numpy())
    train_flow(
        model,
        true_data,
        1000,
        10,
        n_parameters,
        device=device,
    )
