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

true_parameters = 0.4 * torch.ones(
    len(_all_no_seed_parameters)
)  # torch.tensor([0.9, 0.3, 0.6])


class MMDLoss:
    def __init__(self, y):
        self.y = y[0]
        device = self.y.device
        self.y_matrix = self.y.reshape(1, -1, 1)
        self.y_sigma = torch.median(
            torch.pow(torch.cdist(self.y_matrix, self.y_matrix), 2)
        )
        ny = self.y.shape[0]
        self.kyy = (
            torch.exp(
                -torch.pow(torch.cdist(self.y_matrix, self.y_matrix), 2) / self.y_sigma
            )
            - torch.eye(ny, device=device)
        ).sum() / (ny * (ny - 1))

    def __call__(self, x, y):
        nx = x.shape[0]
        x_matrix = x.reshape(1, -1, 1)
        kxx = torch.exp(-torch.pow(torch.cdist(x_matrix, x_matrix), 2) / self.y_sigma)
        # kxx = torch.nan_to_num(kxx, 0.)
        kxx = (kxx - torch.eye(nx, device=device)).sum() / (nx * (nx - 1))
        kxy = torch.exp(
            -torch.pow(torch.cdist(x_matrix, self.y_matrix), 2) / self.y_sigma
        )
        # kxy = torch.nan_to_num(kxy, 0.)
        kxy = kxy.mean()
        return kxx + self.kyy - 2 * kxy


def loss_fn(x, y):
    mask = (x > 0) & (y > 0)
    x = x[mask].log10()
    y = y[mask].log10()
    return torch.nn.MSELoss()(x, y)


def make_flow(n_parameters, device):
    K = 3
    torch.manual_seed(0)
    hidden_units = 128
    hidden_layers = 3
    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                n_parameters, hidden_layers, hidden_units
            )
        ]
        flows += [nf.flows.LULinearPermute(n_parameters)]
    q0 = nf.distributions.DiagGaussian(n_parameters, trainable=False)
    flow = nf.NormalizingFlow(q0=q0, flows=flows)
    return flow.to(device)


def make_flow2(n_parameters, device):
    base = nf.distributions.base.DiagGaussian(n_parameters)
    num_layers = 5
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([n_parameters // 2 + 1, 50, 50, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(n_parameters, mode="swap"))
    flow = nf.NormalizingFlow(base, flows).to(device)
    return flow


def make_prior(n_parameters, device):
    prior = torch.distributions.MultivariateNormal(
        0.0 * torch.ones(n_parameters, device=device),
        1.0 * torch.eye(n_parameters, device=device),
    )
    return prior


def train_flow(model, true_data, n_epochs, n_samples_per_epoch, n_parameters, device):
    torch.manual_seed(0)
    prior = make_prior(n_parameters, device)
    estimator = make_flow2(n_parameters, device)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-3)
    calibrator = Calibrator(
        model=model,
        posterior_estimator=estimator,
        prior=prior,
        data=true_data,
        optimizer=optimizer,
        n_samples_per_epoch=n_samples_per_epoch,
        w=1e-3,
        n_samples_regularisation=10_000,
        forecast_loss=loss_fn,
        log_tensorboard=True,
        gradient_estimation_method="pathwise",
        gradient_horizon=0,
        gradient_clipping_norm=1.0,
        diff_mode="forward",
        device=device,
        jacobian_chunk_size=None,
    )
    calibrator.run(n_epochs=100000, max_epochs_without_improvement=np.inf)
    return calibrator


def make_model(config_file, device):
    config = yaml.safe_load(open(config_file))
    config["system"]["device"] = device
    config[
        "data_path"
    ] = "/cosma7/data/dp004/dc-quer1/gradabm_june_graphs/london_leisure_1.pkl"
    model = June(
        config,
        parameters_to_calibrate=_all_no_seed_parameters,  # ("beta_household", "beta_company", "beta_school"),
    )
    return model


if __name__ == "__main__":
    # parse arguments from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", default=["cpu"], nargs="+")
    args = parser.parse_args()

    # device of this rank
    device = args.device_ids[mpi_rank]
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
        5,
        n_parameters,
        device=device,
    )
