---
title: 'BlackBIRDS: Black-Box Inference foR Differentiable Simulators'
tags:
  - Python
  - Bayesian inference
  - differentiable simulators
  - variational inference
  - Markov chain Monte Carlo
authors:
  - name: Arnau Quera-Bofarull
    orcid: 0000-0001-5055-9863
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Joel Dyer
    orcid: 0000-0002-8304-8450
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2"
  - name: Anisoara Calinescu
    orcid: 0000-0003-2082-734X
    affiliation: 1
  - name: J. Doyne Farmer 
    orcid: 0000-0001-7871-073X
    affiliation: "2, 3, 4"
  - name: Michael Wooldridge
    orcid: 0000-0002-9329-8410
    affiliation: 1
affiliations:
 - name: Department of Computer Science, University of Oxford, UK
   index: 1
 - name: Institute for New Economic Thinking, University of Oxford, UK
   index: 2
 - name: Mathematical Institute, University of Oxford, UK
   index: 3
 - name: Santa Fe Institute, USA
   index: 4
date: 14 July 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`BlackBIRDS` is a Python package consisting of generically applicable, black-box
inference methods for differentiable simulation models. It facilitates both (a) 
the differentiable implementation of simulation models by providing a common 
object-oriented framework for their implementation in `PyTorch` [@pytorch], 
and (b) the use of a variety of gradient-assisted inference procedures for these simulation
models, allowing researchers to easily exploit the differentiable nature of their simulator in 
parameter estimation tasks. The package consists of both Bayesian and non-Bayesian
inference methods, and relies on well-supported software libraries [e.g.,
`normflows`, @normflows] to provide this broad functionality.

# Statement of need

Across scientific disciplines and application domains, simulation is
used extensively as a means to studying complex mathematical models of real-world
systems. A simulation-based approach to modelling such systems provides the
modeller with significant benefits, permitting them to specify their model in the
way that they believe most faithfully represents the true data-generating
process and relieving them from concerns regarding the mathematical tractability
of the resulting model. However, this additional flexibility comes at a price:
the resulting model can be too complex to easily perform optimisation and
inference tasks on the corresponding simulator, which in many cases necessitates 
the use of approximate, simulation-based inference and optimisation methods to perform
these tasks inexactly.

The complicated and black-box nature of many simulation models can present a
significant barrier to the successful deployment of these simulation-based inference 
and optimisation techniques. Consequently, there has been increasing interest within
various scientific communities in constructing *differentiable* simulation models [see e.g., @hep; @gradabm]: 
simulation models for which the gradient of the model output with respect to the
model's input parameters can be easily obtained. The primary motivation for this is
that access to this additional information, which captures the sensitivity of the
output of the simulator to changes in the input, can enable the use of more efficient 
simulation-based optimisation and inference procedures, helping to reduce the total runtime of such 
algorithms, their overall consumption of valuable computational resources, and their 
concomitant financial and environmental costs.

To this end, `BlackBIRDS` was designed to provide researchers with easy access to a
set of parameter inference methods that exploit the gradient information provided by
differentiable simulators. The package provides support for a variety of approaches to
gradient-assisted parameter inference, including:

- Simulated Minimum Distance [SMD, see e.g. @ii; @smm], in which parameter
  point estimates $\hat{\boldsymbol{\theta}}$ are obtained as

  \begin{equation}
      \hat{\boldsymbol{\theta}} 
      = 
      \arg \min_{\boldsymbol{\theta} \in \boldsymbol{\Theta}} {
        \ell(\boldsymbol{\theta}, \mathbf{y})
      },
  \end{equation}
  where $\boldsymbol{\theta}$ are the simulator's parameters, which take values in some
  set $\boldsymbol{\Theta}$, and $\ell$ is a loss function capturing the compatibility between
  the observed data $\mathbf{y}$ and the simulator's behaviour at parameter vector 
  $\boldsymbol{\theta}$;
- Markov chain Monte Carlo (MCMC), in which samples from a parameter posterior 

    \begin{equation}
        \pi(\boldsymbol{\theta} \mid \mathbf{y}) \propto e^{-\ell(\boldsymbol{\theta}, \mathbf{y})} \pi(\boldsymbol{\theta})
    \end{equation}

  corresponding to a choice of loss function $\ell$ and a prior density $\pi$ over $\boldsymbol{\Theta}$ 
  are generated by executing a Markov chain on $\boldsymbol{\Theta}$. Currently, support is 
  provided for Metropolis-adjusted Langevin Dynamics [@mala], although nothing prevents
  the inclusion of additional 
  gradient-assisted MCMC algorithms such as Hamiltonian Monte Carlo [@hmc];
- Variational Inference (VI), in which a parameteric approximation $q^*$ to the
  intractable posterior is obtained by solving the following optimisation problem over a
  variational family $\mathcal{Q}$:
  \begin{equation}
    q^* = \arg\min_{q \in \mathcal{Q}} {
            \mathbb{E}_{q}\left[ -\ell(\boldsymbol{\theta}, \mathbf{y})\right]
            + \mathbb{E}_{q}\left[\log\frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})}\right]
    }
    \end{equation}
  where $\ell$ is defined as above and $\pi$ is a prior density over $\boldsymbol{\Theta}$.

The package is written such that the user is free to specify their choice of $\ell$ and $\pi$ (in the
case of Bayesian methods), under the constraint that both choices are differentiable with respect to
$\boldsymbol{\theta}$. This allows the user to target a wide variety of parameter point estimators, 
and both classical and generalised [see e.g. @bissiri; @gvi] posteriors. We provide a
number of [tutorials](https://www.arnau.ai/blackbirds/) demonstrating (a) how to implement a simulator
in a differentiable framework in PyTorch and (b) how to apply the different parameter inference methods
supported by `BlackBIRDS` to these differentiable simulators. Our package provides the user with flexible
posterior density estimators with the use of normalising flows, and has already been used in scientific
research to calibrate differentiable simulators, such as [@ai4abm; @dae].

## Related software

`BlackBIRDS` offers complementary functionality to a number of existing Python packages. `sbi` [@sbi]
is a package offering PyTorch-based implementations of numerous simulation-based inference algorithms,
including those based on the use of MCMC and neural conditional density estimators. Our package differs
significantly, however: in contrast to `sbi`, `BlackBIRDS` provides support for both Bayesian and 
non-Bayesian inference methods, and permits the researcher to exploit gradients of the simulator, 
loss function, and/or posterior density with respect to parameters $\mathbf{\theta}$ during inference tasks. The same comparison applies to the the `BayesFlow` package [@radev2023bayesflow].
`black-it` [@blit] is a further recent Python package that collects some recently developed parameter 
estimation methods from the agent-based modelling community; the focus of this package is, however, on
non-Bayesian methods, and the package does not currently support the exploitation of simulator gradients.
`PyVBMC` [@pyvbmc] provides a Python implementation of the Variational Bayesian Monte Carlo algorithm 
using Gaussian processes, but differs from our package in that it does not exploit simulator gradients
and is focused on Bayesian inference alone. Additional older packages (e.g. [@abcpy; @pyabc]) also
focus on approximate Bayesian inference methods for non-differentiable simulators. Beyond this, we are
unaware of other mature software projects in Python that support parameter inference in the specific case
of differentiable simulation models. 

# Features

- User-friendly and flexible API: SMD only requires the loss 
  function $\ell$ and the optimiser to use, while MCMC (resp. VI) requires only the loss $\ell$, the prior 
  density $\pi$, and the MCMC method (resp. posterior approximator) to be specified. However, additional arguments
  can be provided to straightforwardly customise hyperparameters of the different methods.
- Multi-GPU parallelisation support with MPI.
- Support for both forward-mode and reverse-mode auto-differentiation.
- Continuous integration and unit tests.

# Acknowledgements

This research was supported by a UKRI AI World Leading Researcher Fellowship awarded to Wooldridge (grant EP/W002949/1). 
M. Wooldridge and A. Calinescu acknowledge funding from Trustworthy AI - Integrating Learning, Optimisation and Reasoning 
([TAILOR](https://tailor-network.eu/)), a project funded by European Union Horizon2020 research and innovation program 
under Grant Agreement 952215.

# References
