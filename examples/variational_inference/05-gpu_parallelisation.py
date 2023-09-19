"""
This scripts shows how to run blackbirds in parallel using MPI4PY.
The parallelization is done across the number of parameters that are sampled
in each epoch from the posterior candidate.

As an example we consider the SIR model.
"""
import argparse
import torch
import networkx
import normflows as nf
import numpy as np

from blackbirds.models.sir import SIR
from blackbirds.infer.vi import VI
from blackbirds.mpi_setup import mpi_rank
from blackbirds.simulate import simulate_and_observe_model


class L2Loss:
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, params, data):
        observed_outputs = simulate_and_observe_model(
            self.model, params, gradient_horizon=0
        )
        return self.loss_fn(observed_outputs[0], data[0])


def make_model(n_agents, n_timesteps, device):
    graph = networkx.watts_strogatz_graph(n_agents, 10, 0.1)
    return SIR(graph=graph, n_timesteps=n_timesteps, device=device)


def make_flow(device):
    # Define flows
    torch.manual_seed(0)
    K = 4
    latent_size = 3
    hidden_units = 64
    hidden_layers = 2

    flows = []
    for _ in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set prior and q0
    q0 = nf.distributions.DiagGaussian(3, trainable=False)

    # Construct flow model
    flow = nf.NormalizingFlow(q0=q0, flows=flows)
    return flow.to(device)


def train_flow(flow, model, true_data, n_epochs, n_samples_per_epoch, device):
    torch.manual_seed(0)
    # Define a prior
    prior = torch.distributions.MultivariateNormal(
        -2.0 * torch.ones(3, device=device), torch.eye(3, device=device)
    )

    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)

    # We set the regularisation weight to 10.
    w = 100

    loss = L2Loss(model)
    # Note that we can track the progress of the training by using tensorboard.
    # tensorboard --logdir=runs
    vi = VI(
        loss=loss,
        posterior_estimator=flow,
        prior=prior,
        optimizer=optimizer,
        w=w,
        n_samples_per_epoch=n_samples_per_epoch,
        device=device,
    )

    # and we run for 500 epochs without early stopping.
    vi.run(true_data, n_epochs=n_epochs, max_epochs_without_improvement=np.inf)


if __name__ == "__main__":
    # parse arguments from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_agents", type=int, default=5000)
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--n_samples_per_epoch", type=int, default=5)
    parser.add_argument("--device_ids", default=["cpu"], nargs="+")
    args = parser.parse_args()

    # device of this rank
    device = args.device_ids[mpi_rank]

    model = make_model(args.n_agents, args.n_timesteps, device)
    true_parameters = torch.tensor(
        [0.05, 0.05, 0.05], device=device
    ).log10()  # SIR takes log parameters
    true_data = model.run_and_observe(true_parameters)
    flow = make_flow(device)
    train_flow(
        flow, model, true_data, args.n_epochs, args.n_samples_per_epoch, device=device
    )
