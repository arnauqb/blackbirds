import torch
import torch_geometric

class SIRMessagePassing(torch_geometric.nn.conv.MessagePassing):
    def forward(self, edge_index, infected, susceptible):
        return self.propagate(edge_index, x=infected, y=susceptible)
    def message(self, x_j, y_i):
        return x_j * y_i 

class SIR(torch.nn.Module):
    def __init__(self, graph, n_timesteps):
        """
        Implements a differentiable SIR model on a graph.

        Arguments:
            graph (networkx.Graph) : a networkx graph
            n_timesteps (int) : the number of timesteps to run the model for
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        # convert graph from networkx to pytorch geometric
        self.graph = torch_geometric.utils.convert.from_networkx(graph)
        self.mp = SIRMessagePassing(aggr='add', node_dim=-1)

    def sample_bernoulli_gs(self, probs, tau=0.1):
        """
        Samples from a Bernoulli distribution in a diferentiable way using Gumble-Softmax
        
        Arguments:
            probs (torch.Tensor) : a tensor of shape (n,) containing the probabilities of success for each trial
            tau (float) : the temperature of the Gumble-Softmax distribution
        """
        logits = torch.vstack((probs, 1 - probs)).T.log()
        gs_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        return gs_samples[:,0]
        

    def forward(self, params):
        """
        Runs the model forward

        Arguments:
            params (torch.Tensor) : a tensor of shape (3,) containing the fraction of infected, beta, and gamma
        """
        # Initialize the parameters
        initial_infected = params[0]
        beta = params[1]
        gamma = params[2]
        n_agents = self.graph.num_nodes
        # Initialize the state
        infected = torch.zeros(n_agents)
        susceptible = torch.ones(n_agents)
        recovered = torch.zeros(n_agents)
        # sample the initial infected nodes
        probs = initial_infected * torch.ones(n_agents)
        new_infected = self.sample_bernoulli_gs(probs)
        infected += new_infected
        susceptible -= new_infected

        infected_hist = infected.sum().reshape((1,))
        recovered_hist = torch.zeros((1,))

        # Run the model forward
        for _ in range(self.n_timesteps):
            # Get number of infected neighbors per node, return 0 if node is not susceptible.
            n_infected_neighbors = self.mp(self.graph.edge_index, infected, susceptible)
            # each contact has a beta chance of infecting a susceptible node
            prob_infection = 1 - (1 - beta) ** n_infected_neighbors
            # sample the infected nodes
            new_infected = self.sample_bernoulli_gs(prob_infection)
            # sample recoverd people
            prob_recovery = gamma * infected
            new_recovered = self.sample_bernoulli_gs(prob_recovery)
            # update the state of the agents
            infected = infected + new_infected - new_recovered
            susceptible -= new_infected
            recovered += new_recovered
            infected_hist = torch.hstack((infected_hist, infected.sum().reshape(1,)))
            recovered_hist = torch.hstack((recovered_hist, recovered.sum().reshape(1,)))

        return infected_hist, recovered_hist

