import torch
import numpy as np
import networkx

from birds.models.sir import SIR


class TestSIR:
    """
    Tests the SIR implementation.
    """

    def test__implementation_against_nnlib(self):
        """
        Compares results with the nnlib implementation.
        https://github.com/GiulioRossetti/ndlib/blob/master/docs/reference/models/epidemics/SIR.rst
        ```python
        import networkx as nx
        import ndlib.models.ModelConfig as mc
        import ndlib.models.epidemics as ep
        import numpy as np

        # Network topology
        g = nx.erdos_renyi_graph(1000, 0.01)
        n_timesteps = 10
        beta = 0.05
        gamma = 0.10
        fraction_infected = 0.05

        # Model selection
        model = ep.SIRModel(g)

        # Model Configuration
        cfg = mc.Configuration()
        cfg.add_model_parameter("fraction_infected", fraction_infected)
        cfg.add_model_parameter('beta', beta)
        cfg.add_model_parameter('gamma', gamma)
        model.set_initial_status(cfg)

        # Simulation execution

        status = np.array(list(model.status.values()))
        infected = (status == 1).sum()
        recovered = 0
        susceptible = N - infected

        iterations = model.iteration_bunch(n_timesteps)

        susc = []
        inf = []
        rec = []
        for iteration in iterations:
            status = np.array(list(iteration["status"].values()))
            new_infected = (status == 1).sum()
            new_recovered = (status == 2).sum()
            infected = infected + new_infected - new_recovered
            recovered += new_recovered
            susceptible -= new_infected
            susc.append(susceptible)
            inf.append(infected)
            rec.append(recovered)

        print(inf)
        print(rec)
        ```
        >>> [50, 100, 114, 135, 174, 217, 260, 304, 329, 377, 407]
        >>> [0, 0, 3, 6, 11, 25, 41, 62, 93, 122, 163]

        """
        N = 1000
        fraction_infected = np.log10(0.05)
        beta = np.log10(0.05)
        gamma = np.log10(0.10)
        n_timesteps = 10  # we do one more...
        graph = networkx.erdos_renyi_graph(N, 0.01)
        model = SIR(graph=graph, n_timesteps=n_timesteps)
        infected, recovered = model(
            torch.tensor([fraction_infected, beta, gamma])
        )  # fraction infected, beta, and gamma
        exp_infected = torch.tensor(
            [50, 100, 114, 135, 174, 217, 260, 304, 329, 377, 407], dtype=torch.float
        )
        exp_recovered = torch.tensor(
            [0, 0, 3, 6, 11, 25, 41, 62, 93, 122, 163], dtype=torch.float
        )
        # check initial infected fraction
        assert np.isclose(infected[0], 0.05 * N, rtol=0.3, atol=2)
        # values from nndlib run
        assert torch.allclose(
            infected[:-1], exp_infected[1:], rtol=1.0
        )  # check is close within a factor of 2... seed is tricky here
        # they do recovery differently...
        assert torch.allclose(recovered[:-1], exp_recovered[1:], rtol=1.0, atol=5)

    def test__gradient_propagates(self):
        """
        Checks that the gradient propagates through the model.
        """
        N = 1000
        beta = -2.0
        gamma = -1.0
        fraction_infected = -2.0
        n_timesteps = 10
        graph = networkx.erdos_renyi_graph(N, 0.01)
        model = SIR(graph=graph, n_timesteps=n_timesteps)
        probs = torch.tensor([fraction_infected, beta, gamma], requires_grad=True)
        infected, _ = model(probs)
        infected.sum().backward()
        assert probs.grad is not None
