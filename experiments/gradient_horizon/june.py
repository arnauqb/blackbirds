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
    #"beta_care_visit",
    "beta_care_home",
]
_all_parameters = ["seed"] + _all_no_seed_parameters

true_parameters = torch.tensor([-3.5, 0.6, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6])
assert len(true_parameters) == len(_all_parameters)
#true_parameters = 0.35 * torch.ones(len(_all_no_seed_parameters))#torch.tensor([0.9, 0.3, 0.6])
#true_parameters = torch.hstack((torch.tensor([-3.5]), true_parameters))
#true_parameters = torch.tensor([-3.5, 0.9, 0.3, 0.6])

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

def loss_fn2(x, y):
    mask = y > 0
    x = x[mask]
    y = y[mask]
    return ((x - y)**2 / y**2).mean()


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
        param_map = nf.nets.MLP([n_parameters//2 + n_parameters % 2 , 50, 50, n_parameters], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(n_parameters, mode="swap"))
    flow = nf.NormalizingFlow(base, flows).to(device)
    return flow

def make_flow3(n_parameters, device):
    # Define flows
    K = 5
    torch.manual_seed(0)

    latent_size = n_parameters
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 4 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 4 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]
    q0 = nf.distributions.DiagGaussian(n_parameters)

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm.to(device)

def make_flow4(n_parameters, device):
    K = 16
    torch.manual_seed(0)
    flows = []
    for i in range(K):
        flows.append(nf.flows.MaskedAffineAutoregressive(n_parameters, 20, num_blocks=2))
        flows.append(nf.flows.Permute(n_parameters, mode="swap"))
    q0 = nf.distributions.DiagGaussian(n_parameters)
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm.to(device)


def make_prior(n_parameters, device):
    means = torch.hstack((torch.tensor([-3.]), 0.0 * torch.ones(n_parameters-1)))
    #means = 0.0 * torch.ones(n_parameters)
    means = means.to(device)
    prior = torch.distributions.MultivariateNormal(
        means,
        0.5 * torch.eye(n_parameters, device=device),
    )
    return prior


def train_flow(model, true_data, n_epochs, n_samples_per_epoch, n_parameters, device):
    torch.manual_seed(0)
    prior = make_prior(n_parameters, device)
    estimator = make_flow4(n_parameters, device)
    #estimator.load_state_dict(torch.load("./london_1.pt"))
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-3)
    calibrator = Calibrator(
        model=model,
        posterior_estimator=estimator,
        prior=prior,
        data=true_data,
        optimizer=optimizer,
        initialize_flow_to_prior=False,
        n_samples_per_epoch=n_samples_per_epoch,
        w=1e-4,
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
        #] = "/cosma7/data/dp004/dc-quer1/gradabm_june_graphs/london_leisure_1.pkl"
        ] = "/Users/arnull/code/gradabm-june/worlds/london_leisure_1.pkl"
    model = June(
        config,
        parameters_to_calibrate=_all_parameters, #("seed", "beta_household", "beta_company", "beta_school"),
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
    to_save = np.array([data.cpu().numpy() for data in true_data])
    np.savetxt("./true_data.txt", to_save)
    train_flow(
        model,
        true_data,
        1000,
        5,
        n_parameters,
        device=device,
    )
