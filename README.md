![](docs/_static/banner.png)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.05776/status.svg)](https://doi.org/10.21105/joss.05776)
[![codecov](https://codecov.io/gh/arnauqb/blackbirds/branch/main/graph/badge.svg?token=HvwGGjA7qr)](https://codecov.io/gh/arnauqb/blackbirds)
[![Docs](https://github.com/arnauqb/blackbirds/actions/workflows/docs.yml/badge.svg)](https://arnau.ai/blackbirds)
[![Build and test package](https://github.com/arnauqb/birds/actions/workflows/ci.yml/badge.svg)](https://github.com/arnauqb/birds/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/blackbirds.svg)](https://badge.fury.io/py/blackbirds)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


`BlackBIRDS` is a Python package consisting of generically applicable, black-box inference methods for differentiable simulation models. It facilitates both (a) the differentiable implementation of simulation models by providing a common object-oriented framework for their implementation in PyTorch, and (b) the use of a variety of gradient-assisted inference procedures for these simulation models, allowing researchers to easily exploit the differentiable nature of their simulator in parameter estimation tasks. The package consists of both Bayesian and non-Bayesian inference methods, and relies on well-supported software libraries (e.g. normflows, Stimper et al. (2023)) to provide this broad functionality.


# 1. Installation

The easiest way to install Birds is to install it from the PyPI repository

```
pip install blackbirds
```

To get the latest development version, you can install it directly from git

```
pip install git+https://github.com/arnauqb/blackbirds
```

# 2. Usage

Refer to the [docs](https://arnau.ai/blackbirds) for examples and specific API usage. Here is a basic example:

```python
import torch

from blackbirds.models.random_walk import RandomWalk
from blackbirds.infer.vi import VI
from blackbirds.posterior_estimators import TrainableGaussian
from blackbirds.simulate import simulate_and_observe_model

# random walk model
rw = RandomWalk(n_timesteps=100)

# generate synthetic data to fit to
true_p = torch.logit(torch.tensor(0.25))
true_data = rw.observe(rw.run(torch.tensor([true_p])))

# define loss to minimize
class L2Loss:
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
    def __call__(self, params, data):
        observed_outputs = simulate_and_observe_model(self.model, params)
        return self.loss_fn(observed_outputs[0], data[0])

# initialize generalized variational inference
posterior_estimator = TrainableGaussian([0.], 1.0)
prior = torch.distributions.Normal(true_p + 0.2, 1)
optimizer = torch.optim.Adam(posterior_estimator.parameters(), 1e-2)
loss = L2Loss(rw)

vi = VI(loss,
        posterior_estimator=posterior_estimator,
        prior=prior,
        optimizer=optimizer,
        w = 0) # no regularization

# train the estimator
vi.run(true_data, 1000, max_epochs_without_improvement=100)

```

# 3. Tests

To run the unit tests of the code, you need to have pytest installed,

```bash
pip install pytest pytest-cov
```

and run the command

```bash
pytest test
```

# 4. Contributing

See [CONTRIBUTING.md](https://github.com/arnauqb/blackbirds/blob/main/CONTRIBUTING.md) for the contribution guidelines.

# 5. Citation

```
@article{Quera-Bofarull2023, 
    doi = {10.21105/joss.05776}, 
    url = {https://doi.org/10.21105/joss.05776}, 
    year = {2023}, 
    publisher = {The Open Journal}, 
    volume = {8}, 
    number = {89}, 
    pages = {5776}, 
    author = {Arnau Quera-Bofarull and Joel Dyer and Anisoara Calinescu and J. Doyne Farmer and Michael Wooldridge}, 
    title = {BlackBIRDS: Black-Box Inference foR Differentiable Simulators}, 
    journal = {Journal of Open Source Software} }
```
