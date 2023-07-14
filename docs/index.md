# BlackBIRDS documentation

BlackBIRDS is a Python package consisting of generically applicable, black-box inference methods for differentiable simulation models. It facilitates both (a) the differentiable implementation of simulation models by providing a common object-oriented framework for their implementation in PyTorch, and (b) the use of a variety of gradient-assisted inference procedures for these simulation models, allowing researchers to easily exploit the differentiable nature of their simulator in parameter estimation tasks. The package consists of both Bayesian and non-Bayesian inference methods, and relies on well-supported software libraries (e.g. normflows, Stimper et al. (2023)) to provide this broad functionality.

## Installation

The easiest way to install the package is to obtain it from the PyPI repository

```
pip install blackbirds
```

Alternatively, you can obtain the latest version directly from git, 

```
pip install git+https://github.com/arnauqb/blackbirds
```
