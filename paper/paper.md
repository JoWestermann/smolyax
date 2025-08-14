---
title: 'smolyax: a high-performance implementation of the Smolyak interpolation operator in JAX'
tags:
  - Polynomial Interpolation
  - Smolyak Operator
  - Sparse Grids
  - Polynomial Chaos
  - JAX
  - Numba
authors:
  - name: Josephine Westermann
    orcid: 0000-0003-3450-9166
    affiliation: 1
    corresponding: true
  - name: Joshua Chen
    orcid: 0009-0002-2257-5780
    affiliation: 2
affiliations:
 - name: Heidelberg University, Germany
   ror: 038t36y30
   index: 1
 - name: Colorado State University, USA
   ror: 03k1gpj17
   index: 2
date: 28 February 2025
bibliography: paper.bib
header-includes:
  - \usepackage{amssymb}
  - \usepackage{algorithm, algpseudocode}
  - \usepackage{bm}
  - \newcommand{\R}{\mathbb{R}}
  - \newcommand{\N}{{\mathbb N}}
  - \newcommand{\bbP}{{\mathbb P}}
  - \newcommand{\set}[2]{\{#1\,:\,#2\}}
  - \newcommand{\bsb}{{\pmb{b}}}
  - \newcommand{\bse}{{\pmb{e}}}
  - \newcommand{\bsk}{{\pmb{k}}}
  - \newcommand{\bsx}{{\pmb{x}}}
  - \newcommand{\bsF}{{\pmb{F}}}
  - \newcommand{\bsT}{{\pmb{T}}}
  - \newcommand{\bsmu}{{\pmb{\mu}}}
  - \newcommand{\bsnu}{{\pmb{\nu}}}
  - \newcommand{\bstau}{{\pmb{\tau}}}
  - \newcommand{\bsrho}{{\pmb{\rho}}}
  - \newcommand{\bsxi}{{\pmb{\xi}}}
---

# Summary

The `smolyax` library provides interpolation capabilities for arbitrary multivariate and vector-valued functions $f : \mathbb{R}^{d_{\rm in}} \to \mathbb{R}^{d_{\rm out}}$ for any $d_{\rm in}, d_{\rm out} \in \mathbb{N}$. It implements the Smolyak interpolation operator, which is known to overcome the curse-of-dimensionality plaguing naive multivariate interpolation [@barthelmann:2000] and uses the barycentric interpolation formula [@berrut:2004] for numerical stability. The implementation is based on JAX [@jax:2018], a free and open-source Python library for high-performance computing that integrates with the Python and NumPy numerical computing ecosystem. Thanks to JAX's device management, `smolyax` runs natively on both CPU and GPU. While implementing Smolyak interpolation in JAX is challenging due to the highly irregular data structures involved, `smolyax` overcomes this by employing a tailored batching and padding strategy that enables efficient vectorization.

`smolyax` supports interpolation on bounded or unbounded domains via Leja [@chkifa:2013] or Gauss-Hermite [@abramowitz:1964] node sequences, respectively. It provides efficient Numba-accelerated routines to generate isotropic or anisotropic total degree multi-index sets [@adcock:2022, §2.3.2], which are the key ingredient to generate the high-dimensional sparse grids [@bungartz:2004] of interpolation nodes required by the Smolyak interpolation operator. Additional types of node sequences or multi-index sets can be incorporated easily by implementing a minimalistic interface. The `smolyax` interpolant provides further functionality to evaluate its gradient as well as compute its integral.

# Statement of Need

Polynomial expansion is a well-studied and powerful tool in applied mathematics, with important applications in surrogate modeling, uncertainty quantification and inverse problems, see e.g., @adcock:2022, @dung:2023, @zech:2018, @chkifa:2015, @herrmann:2024, @westermann:2025, and references therein. Smolyak interpolation offers a practical way to construct polynomial approximations with known error bounds for a wide range of function classes, see e.g., @barthelmann:2000, @chkifa:2015, and @adcock:2022.

Several libraries provide CPU-based high-dimensional interpolation functionality, for example `Chaospy` [@feinberg:2015], `UQLab` [@marelli:2014], `The Sparse Grid Matlab Kit` [@piazzola:2024], `PyApprox` [@jakeman:2023], `MUQ` [@parno:2021], and `UncertainSCI` [@tate:2023]. The GPU support that is necessary in practice to go from moderate to high dimensions is offered so far only by `Tasmanian` [@tasmanian]. Benchmark experiments suggest that while asymptotic runtime of the Smolyak interpolator in `Tasmanian` scale better as the output dimensions $d_{\rm out}$ increases, `smolyax` offers competitive asymptotic runtimes for increasing $d_{\rm in}$ and input data set size.

# A vectorizable implementation of the Smolyak operator

Recall that given a domain $D \subset \R$ and set of $\nu + 1 \in \N$ pairwise distinct interpolation points $(\xi^\nu_i)_{i=0}^\nu \subset D$, the univariate polynomial interpolation operator $I^\nu$ maps a function $f : D \to \R$ onto the unique polynomial $I^\nu [f] \in \bbP_\nu := {\rm span} \set{x^i}{i=0,\dots,\nu}$ of maximal degree $\nu$ such that $f(\xi^\nu_i) = I^\nu [f](\xi^\nu_i)$ for all $i\in\{0,1,\dots,\nu\}$.

Tensorized interpolation generalizes univariate interpolation to multivariate functions defined on a tensor-product domain $D = \otimes_{j=1}^d D_1$ with $D_1 \subset \R$ and $d \in \N$ by defining $I^\bsnu := \otimes_{j=1}^d I^{\nu_j}$, where $\bsnu \in \N_0^d$ is a multi-index characterizing the maximal polynomial degree in each dimension.
Since $I^\bsnu[f] \in \bbP_\bsnu := {\rm span} \set{\bsx^\bsmu}{\bsmu \leq \bsnu}$, this approach suffers from the curse of dimensionality as $d$ increases.

Smolyak interpolation [@smolyak:1963; @barthelmann:2000] overcomes this issue by introducing polynomial ansatz spaces $\bbP_\Lambda := {\rm span} \set{\bsx^\bsmu}{\bsmu \in \Lambda}$ parametrized by downward closed multi-index sets $\Lambda \subset \N_0^d$. The resulting interpolation operator is a linear combination of tensorized interpolation operators:
\begin{equation*} \label{eq:ip_smolyak}
    I^\Lambda := \sum \limits_{\bsnu \in \Lambda} \zeta_{\Lambda, \bsnu} I^\bsnu, \qquad \zeta_{\Lambda, \bsnu} := \sum \limits_{\bse \in \{0,1\}^d : \bsnu+\bse \in \Lambda} (-1)^{|\bse|}.
\end{equation*}

Implementing this operator in a vectorized form suitable for high-performance computing is nontrivial, as vectorization requires inputs to conform to a uniform structure. However, each tensorized interpolant in the equation above involves reducing a higher-order tensor of unique shape $\bsnu$ via multiplication with one vector per dimension. A naive strategy would be to zero-pad all tensors in this equation to the smallest common shape $(\max_{\bsnu \in \Lambda}(\nu_j))_{j=1}^d$. This, however, reintroduces the curse of dimensionality, as memory requirements grow exponentially with $d$.

`smolyax` strikes a balance between handling small, uniquely shaped tensors and large, identically shaped ones. The key idea is to group tensors by their number of active dimensions and prepare them for vectorized processing within each group. In particular, this involves
\begin{itemize}
\item[1.] Dropping indices ("\textit{squeezing}") of non-active dimensions, i.e., $j$ with $\nu_j = 0$.
\item[2.] Permuting the remaining active dimensions in descending order, and
\item[3.] Zero-padding all tensors with the same number of active dimensions to the smallest common shape.
\end{itemize}
This reorganizes the tensors into a few large, structured blocks enabling fast vectorized processing. Asymptotically, in both dimension and size of the polynomial space, the method requires only a logarithmic-factor increase in overall memory compared with the raw tensors.

# Acknowledgements

We thank Thomas O'Leary-Roseberry and Jakob Zech for insightful discussions in the early stages of this project.

# References
