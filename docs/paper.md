---
title: 'BlackBirds: A Python package for Black-Box Inference for Differentiable Simulators'
tags:
  - Python
  - agent-based models
  - automatic differentiation
  - variational inference
  - bayesian statistics
authors:
  - name: Arnau Quera-Bofarull
    orcid: 0000-0001-5055-9863
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Joel Dyer
    orcid: 
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1

affiliations:
 - name: Department of Computer Science, University of Oxford, UK
   index: 1

date: 12 July 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Computer simulations are used across a wide range of scientific disciplines to model complex systems. These numerical simulators often depend on sets of parameters that need to be calibrated according to some metrics of interest to the problem at hand. 

# Statement of need

`BlackBIRDS` is a package that enables the use of gradient-assisted calibration techniques, such as Simulated Minimum Distance, Variational Inferece, or Langevin Monte Carlo, among others. The package integrates well with PyTorch autograd engine so that gradients obtained through automatic differentiation can be easily used. 

 Multiple existing packages provide functionality for simulation-based inference ([@sbi; @elfi; @carl; @pyabc]).
 
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
