![](docs/_static/banner.png)

[![Docs](https://github.com/arnauqb/blackbirds/actions/workflows/docs.yml/badge.svg)](https://arnau.ai/blackbirds)
[![Build and test package](https://github.com/arnauqb/birds/actions/workflows/ci.yml/badge.svg)](https://github.com/arnauqb/birds/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/arnauqb/blackbirds/branch/main/graph/badge.svg?token=HvwGGjA7qr)](https://codecov.io/gh/arnauqb/blackbirds)

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



# 3. Citation

TODO.
