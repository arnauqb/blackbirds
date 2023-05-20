import torch
import networkx
import torch_geometric

from birds.utils import soft_minimum, soft_maximum
from birds.models import Model


class SIR(Model):
    def __init__(self, graph: networkx.Graph, n_timesteps: int):
        """
        Implements a differentiable SIR model on a graph.

        **Arguments:**

        - graph: a networkx graph
        - n_timesteps: the number of timesteps to run the model for
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        # convert graph from networkx to pytorch geometric
        self.graph = torch_geometric.utils.convert.from_networkx(graph)
        self.mp = SIRMessagePassing(aggr="add", node_dim=-1)

    def sample_bernoulli_gs(self, probs: torch.Tensor, tau: float = 0.1):
        """
        Samples from a Bernoulli distribution in a diferentiable way using Gumble-Softmax

        **Arguments:**

        - probs: a tensor of shape (n,) containing the probabilities of success for each trial
        - tau: the temperature of the Gumble-Softmax distribution
        """
        logits = torch.vstack((probs, 1 - probs)).T.log()
        gs_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        return gs_samples[:, 0]

    def trim_time_series(self, x):
        return x[-1:]

    def initialize(self, params: torch.Tensor):
        """
        Initializes the model setting the adequate number of initial infections.

        **Arguments**:

        - params: a tensor of shape (3,) containing the **log10** of the fraction of infected, beta, and gamma
        """
        params = soft_minimum(params, torch.tensor(0.0, device=params.device), 2)
        params = 10**params
        # set initial fraction of infected
        initial_infected = params[0]
        n_agents = self.graph.num_nodes
        # sample the initial infected nodes
        probs = initial_infected * torch.ones(n_agents)
        new_infected = self.sample_bernoulli_gs(probs)
        # set the initial state
        infected = new_infected
        susceptible = 1 - new_infected
        recovered = torch.zeros(n_agents)
        x = torch.vstack((infected, susceptible, recovered))
        return x.reshape(1, 3, n_agents)

    def step(self, params: torch.Tensor, x: torch.Tensor):
        """
        Runs the model forward for one timestep.

        **Arguments**:

        - params: a tensor of shape (3,) containing the **log10** of the fraction of infected, beta, and gamma
        - x: a tensor of shape (3, n_agents) containing the infected, susceptible, and recovered counts.
        """
        params = soft_minimum(params, torch.tensor(0.0, device=params.device), 2)
        params = 10**params
        _, beta, gamma = params
        infected, susceptible, recovered = x[-1]
        # Get number of infected neighbors per node, return 0 if node is not susceptible.
        n_infected_neighbors = self.mp(self.graph.edge_index, infected, susceptible)
        # each contact has a beta chance of infecting a susceptible node
        n_infected_neighbors = torch.clip(n_infected_neighbors, min=0.0, max=5.0)
        prob_infection = 1 - (1 - beta) ** n_infected_neighbors
        prob_infection = torch.clip(prob_infection, min=1e-10, max=1.0)
        # sample the infected nodes
        new_infected = self.sample_bernoulli_gs(prob_infection)
        # sample recoverd people
        prob_recovery = gamma * infected
        prob_recovery = torch.clip(prob_recovery, min=1e-10, max=1.0)
        new_recovered = self.sample_bernoulli_gs(prob_recovery)
        # update the state of the agents
        infected = infected + new_infected - new_recovered
        susceptible = susceptible - new_infected
        recovered = recovered + new_recovered
        x = torch.vstack((infected, susceptible, recovered)).reshape(1, 3, -1)
        return x

    def observe(self, x: torch.Tensor):
        """
        Returns the total number of infected and recovered agents per time-step

        **Arguments**:

        - x: a tensor of shape (3, n_agents) containing the infected, susceptible, and recovered counts.
        """
        return [x[:, 0, :].sum(1), x[:, 2, :].sum(1)]


class SIRMessagePassing(torch_geometric.nn.conv.MessagePassing):
    """
    Class used to pass messages between agents about their infected status.
    """

    def forward(
        self,
        edge_index: torch.Tensor,
        infected: torch.Tensor,
        susceptible: torch.Tensor,
    ):
        """
        Computes the sum of the product between the node's susceptibility and the neighbors' infected status.

        **Arguments**:

        - edge_index: a tensor of shape (2, n_edges) containing the edge indices
        - infected: a tensor of shape (n_nodes,) containing the infected status of each node
        - susceptible: a tensor of shape (n_nodes,) containing the susceptible status of each node
        """
        return self.propagate(edge_index, x=infected, y=susceptible)

    def message(self, x_j, y_i):
        return x_j * y_i
