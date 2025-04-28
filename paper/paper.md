---
title: 'jax-smolyak: A HPC-capable implementation of the Smolyak interpolation operator'
tags:
  - Python
  - JAX
  - Interpolation
  - HPC
  - Smolyak
  - Sparse Grids
  - Polynomial Chaos
authors:
  - name: Josephine Westermann
    orcid: 0000-0003-3450-9166
    affiliation: 1
    corresponding: true
  - name: Joshua Chen
    affiliation: 2
affiliations:
 - name: Heidelberg University, Germany
   ror: 038t36y30
   index: 1
 - name: Institution Name, Country
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

The `jax-smolyak` library provides interpolation capabilities for arbitrary multivariate and vector-valued functions $f : \mathbb{R}^{d_1} \to \mathbb{R}^{d_2}$ for any $d_1, d_2 \in \mathbb{N}$.

It is based on JAX, a free and open-source Python library for high-performance computing, and integrates seamlessly with the Python ecosystem. Thanks to JAX's device management, `jax-smolyak` runs natively on both CPU and GPU. While implementing Smolyak interpolation in JAX is challenging due to the highly irregular data structures involved, `jax-smolyak` overcomes this by employing a tailored batching and padding strategy (described below), enabling efficient vectorization, scalability, and parallel execution.

`jax-smolyak` supports interpolation on sparse grids based on either Leja or Gauss-Hermite interpolation nodes and characterized by anisotropic multi-index sets of the form
$$
\Lambda_{\mathbf{k}, \ell} := \{\boldsymbol{\nu} \in \mathbb{N}_0^{d_1} : \sum_{j=1}^{d_1} k_j \nu_j < \ell\},
$$
where $\mathbf{k} \in \mathbb{R}^{d_1}$. In the special case $\mathbf{k} = (1)_{j=1}^{d_1}$, this reduces to the classical total-degree multi-index set. Additional types of interpolation nodes or multi-index sets can be incorporated easily by implementing a minimalistic interface.

# Statement of Need

Polynomial approximation is a well-studied and powerful tool in applied mathematics, with important applications, for example, in surrogate modeling and uncertainty quantification. 
Due to their deterministic construction and the availability of error bounds for a wide range of function classes, polynomial surrogates can serve as both a reliable and cost-effective alternative to neural networks — for example, in constructing operator surrogates [@westermann:2025].

While several libraries provide high-dimensional interpolation functionality, none, to our knowledge, provides a hardware-agnostic, high performance implementation. `jax-smolyak` addresses this gap by providing an efficient solution within the popular JAX ecosystem.

# High-dimensional interpolation with the Smolyak operator

We briefly summarize the essentials of high-dimensional interpolation, where sparse grids have become the standard choice for interpolation points. In this setting, the interpolation operator is commonly referred to as the Smolyak operator. This overview provides background and establishes notation, which will be used to describe our specific implementation choices in the next section.

**Univariate interpolation.** Given a domain $D \subset \R$ and set of $\nu \in \N$ pairwise distinct interpolation points $(\xi^\nu_i)_{i=0}^\nu \subset D$, the polynomial interpolation operator $I^\nu : C^0(D) \to \bbP_\nu := {\rm span} \set{x^i}{i=0,\dots,\nu}$ maps a function $f : D \to \R$ onto the unique polynomial $I^\nu [f]$ of maximal degree $\nu$ such that $f(\xi^\nu_i) = I^\nu [f](\xi^\nu_i)$ for all $i\in\{0,1,\dots,\nu\}$.

A classical method for evaluating the interpolating polynomial that is known for numerical stability is barycentric interpolation [@berrut:2004]. The univariate barycentric interpolation formula is given as
\begin{align}
    I^\nu [f] (x) := \frac{\sum_{i=0}^\nu b_i^\nu(x) f(\xi^\nu_i)}{\sum_{i=0}^\nu b_i^\nu(x)},
    \qquad \text{ with }
    b_i^\nu(x) &:= \begin{cases} 1 &\text{ if } i = \nu = 0 \\ \frac{w_i^\nu}{x-\xi^\nu_i} &\text{ else }\end{cases} \\
    \quad \text{ and }
    w_i^\nu &:= \prod \limits_{k\in\{0,1,...,\nu\}/\{i\}}\frac{1}{(\xi^\nu_i - \xi^\nu_k)}.
\end{align}

**Tensorized interpolation** generalizes univariate interpolation to multivariate functions defined on a tensor-product domain $D = \otimes_{j=1}^d D_1$ with $D_1 \subset \R$ and $d \in \N$.
Given a multi-index $\bsnu \in \N_0^d$ characterizing the maximal polynomial degree in each dimension, we define $I^\bsnu : C^0(D) \to \bbP_\bsnu := {\rm span} \set{\bsx^\bsmu}{\bsmu \leq \bsnu}$ as
\begin{equation}
    I^\bsnu := \otimes_{j=1}^d I^{\nu_j}.
\end{equation}

The tensorized barycentric interpolation formula can be expressed as a repeated vector-tensor product
\begin{equation} \label{eq:ip_tensorproduct}
    I^\bsnu [f] (\bsx) =
    \frac{\bsb^{\nu_1}(x_1) \bsb^{\nu_2}(x_2) \cdots \bsb^{\nu_d}(x_d) \bsF^\bsnu}
    {\prod_{j=1}^d (\sum_{i=0}^{\nu_j} b_i^{\nu_j} (x_j))},
\end{equation}
with tensors $\bsF^\bsnu \in \R^{\bsnu+{\bm 1}} := \R^{(\nu_1+1) \times (\nu_2+1) \times \dots \times (\nu_d+1)}$ given as
\begin{align*}
  F^\bsnu_{\bsmu} := f(\bsxi^\bsnu_\bsmu)  \text{ with }
   \bsxi^\bsnu_\bsmu := \left(\xi^{\nu_1}_{\mu_1}, \xi^{\nu_2}_{\mu_2}, \dots, \xi^{\nu_d}_{\mu_d} \right) \quad \text{ for all }
   \bsmu \in \N_0^d \text{ s.t. } \bsmu \le \bsnu,
\end{align*}
and vectors $\bsb^{\nu_j} (x_j) \in \R^{\nu_j+1}$ given as
\begin{equation*}
  \bsb^{\nu_j}(x_j) := (b_i^{\nu_j}(x_j))_{i=0}^{\nu_j} \quad \text{ for all }
  j \in \{1, \dots, d\}.
\end{equation*}

**Smolyak interpolation** [@smolyak:1963; @barthelmann:2000; @adcock:2022] overcomes the curse-of-dimensionality that characterizes tensor-product interpolation \eqref{eq:ip_tensorproduct}, by introducing polynomial ansatz spaces $\bbP_\Lambda := {\rm span} \set{\bsx^\bsmu}{\bsmu \in \Lambda}$ parametrized by downward closed multi-index sets $\Lambda \subset \N_0^d$. The resulting interpolation operator is then a linear combination of tensorized interpolation operators:
\begin{equation} \label{eq:ip_smolyak}
    I^\Lambda := \sum \limits_{\bsnu \in \Lambda} \zeta_{\Lambda, \bsnu} I^\bsnu, \qquad \zeta_{\Lambda, \bsnu} := \sum \limits_{\bse \in \{0,1\}^d : \bsnu+\bse \in \Lambda} (-1)^{|\bse|}.
\end{equation}

# Vectorizable implementation of the Smolyak operator for HPC

To leverage key HPC techniques such as vectorization, parallelization, and batch processing, input data must conform to a uniform structure. However, the vectors and tensors in \eqref{eq:ip_smolyak} together with \eqref{eq:ip_tensorproduct} can exhibit a wide range of shapes, posing a challenge for efficient vectorization. A naive approach would be to zero-pad all tensors $\bsF^\bsnu$ in \eqref{eq:ip_smolyak} to the smallest possible common shape $(\max_{\bsnu \in \Lambda}(\nu_j))_{j=1}^d$. This approach, however, suffers from the curse of dimensionality, as memory requirements grow exponentially with $d$. With our implementation we navigate in between these two extremes of handling a large number of small tensors and a single, massive tensor. The key idea is to set up all tensors by:
\begin{itemize}
\item[1.] Dropping indices ("\textit{squeezing}") $j$ of non-active dimensions, i.e., those with $\nu_j = 0$,
\item[2.] Permuting the remaining active dimensions in descending order, and
\item[3.] Zero-padding all tensors with the same number of active dimensions to the smallest common shape.
\end{itemize}
This reorganizes the tensors into a small number of large, structured data blocks, enabling efficient computation while keeping memory overhead modest.

**Squeezing and permuting the dimensions of the tensorized interpolator.**
For any multi-index $\bsnu \in \N_0^d$, denote with $s(\bsnu)$ the tuple consisting of the elements in $\{j \in \{1, ..., d\} \ : \ \nu_j > 0\}$ that permute the non-zero entries of $\bsnu$ descendingly. Let further $\bsnu^s := (\nu_j)_{j \in s(\bsnu)}$ and $d_\bsnu := |\bsnu_s| \equiv |s(\bsnu)| = | \set{j \in \{1, ..., d\}}{\nu_j > 0}| \le d$.

_Example: For the multi-index $\bsnu = (3,0,2,0,4) \in \N_0^5$, it holds that $s(\bsnu) = (5,1,3)$, $\bsnu^s = (4,3,2)$ and $d_\bsnu = 3$._

With the above notation we can re-express the tensorized interpolation operator for $\bsnu$ as
\begin{equation} \label{eq:ip_truncating}
  I^\bsnu [f] (\bsx) \equiv I^{\bsnu, s} [f] (\bsx) :=
    \frac{\bsb^{\nu^s_1}(x_{s_1(\bsnu)}) \bsb^{\nu^s_2}(x_{s_2(\bsnu)}) \cdots \bsb^{\nu^s_{d_\bsnu}}(x_{s_{d_\bsnu}(\bsnu)}) \bsF^{\bsnu,s}}
    {\prod_{j=1}^{d_\bsnu} (\sum_{i=0}^{\nu^s_j} b_i^{\nu^s_j} (x_{s_j(\bsnu)}))}
    \quad \forall \bsx \in D,
\end{equation}
with $\bsF^{\bsnu,s} \in \R^{\bsnu^s + {\bm 1}}$ given as $F^{\bsnu,s}_{\bsmu} := f(\bsxi^{\bsnu, s}_\bsmu)$ for all $\bsmu \in \N_0^{d_\bsnu}$ s.t. $\bsmu \le \bsnu^s$, with
\begin{align*}
    \left(\bsxi^{\bsnu, s}_\bsmu\right)_j :=
    \begin{cases}
    \xi^{\nu_j}_{\mu_i} &\text{ if } j \in s(\bsnu) \text{ and } j=s_i(\bsnu)\\
    \xi^{\nu_j}_0 &\text{ else.}
    \end{cases}
\end{align*}
This construction is equivalent to \eqref{eq:ip_tensorproduct} since for any dimension $j$ with $\nu_j = 0$ it holds that the corresponding weight vector $\bsb^{\nu_j}(x_j)$ is a one-dimensional vector with entry $1$, i.e. $\bsb^{\nu_j}(x_j) = (1) \in \R^1$.

**Padding tensorized interpolators to a common shape.** For any multi-index $\bstau \in \N_0^k$, define the padding operator $p^\bstau$ that acts on any order-$k$ tensor $\bsT \in \R^{\bsrho}$ with $\bsrho \le \bstau$ as
$$p^\bstau(\bsT) \in \R^{\bstau}, \ p^\bstau(\bsT)_{\bsmu} :=
\begin{cases}
    T_{\bsmu} &\text{ if } \bsmu \le \bsrho\\
    0 &\text{ else.}
\end{cases} \quad \forall \bsmu \le \bstau.$$

_Example: For $\bsrho = (2,3)$, the tensor $\bsT = \begin{pmatrix} 0 & 3 & -1 \\ 2 & 0 & 4 \end{pmatrix} \in \R^\bsnu$ can be padded to $\bstau = (3,5)$ as $$p^\bstau(\bsT) = \begin{pmatrix} 0 & 3 & -1 & 0 & 0 \\ 2 & 0 & 4 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0\end{pmatrix} \in \R^\bstau.$$_

Any tensorized interpolation operator $I^\bsnu$ can be padded to higher dimensions $\bstau \ge \bsnu$ via
\begin{equation} \label{eq:ip_padding}
    I^\bsnu [f] (\bsx) \equiv I^{\bsnu, \bstau}[f] (\bsx) :=
    \frac{p^{\tau_1}(\bsb^{\nu_1}(x_1)) p^{\tau_2}(\bsb^{\nu_2}(x_2)) \cdots p^{\tau_d}(\bsb^{\nu_d}(x_d)) p^\bstau(\bsF^\bsnu)}
    {\prod_{j=1}^d (\sum_{i=0}^{\tau_j} p^{\tau_j}_i(\bsb^{\nu_j} (x_j)))}.
\end{equation}

We write $I^{\bsnu, s, \bstau}$ when applying \eqref{eq:ip_truncating} and \eqref{eq:ip_padding} successively.

**Pseudocode of our implementation strategy.**
We now have everything in place to construct the Smolyak interpolant in a form that is well-suited for vectorized execution. Algorithm \ref{alg:smolyak} outlines the key steps. Given a downward-closed but otherwise arbitrarily structured multi-index set $\Lambda$ and a target function $f$, we begin by identifying the subset $\Lambda_\zeta$ of multi-indices $\bsnu$ with nonzero Smolyak coefficients $\zeta_{\Lambda, \bsnu}$ and determining the maximal number $N \le d$ of nonzero entries across these multi-indices. Notably, $N$ remains small (typically single-digit) even when the dimensionality $d$ reaches the hundreds and $|\Lambda|$ is on the order of tens of thousands. For each fixed sparsity level $n \in [0, \dots, N]$, we extract the subset $\Lambda_n$ of multi-indices with exactly $n$ nonzero entries and determine the minimal bounding multi-index $\bstau \in \mathbb{N}_0^n$ such that $\bsnu^s \leq \bstau$ for all $\bsnu \in \Lambda_n$. This step ensures that all the tensorized interpolation operators $(I^\bsnu)_{\bsnu \in \Lambda_n}$ can be efficiently assembled into a single, vectorized computation. In Algorithm \ref{alg:smolyak}, this is compactly expressed as the summation of all $(I^{\bsnu, s, \bstau})_{\bsnu \in \Lambda_n}$, but in practice, it corresponds to pre-allocating and incrementally populating large arrays for interpolation nodes, weights, and function values. The final interpolant $I^\Lambda$ is then assembled through a brief loop over a small number of high-throughput operations, ensuring computational efficiency.

\begin{algorithm}[H]
  \caption{Construct the multivariate barycentric Smolyak interpolator $I^\Lambda$\\
    \textit{Input:} Target function $f$, multi-index set $\Lambda \subset \N_0^d$\\
    \textit{Output:} $I^\Lambda$} \label{alg:smolyak}
  \begin{algorithmic}[1]
    \State $I^\Lambda = 0$
    \State $\Lambda_\zeta = \set{\bsnu \in \Lambda}{\zeta_{\Lambda, \bsnu} \neq 0}$
    \State $N = \max \set{d_\bsnu}{\bsnu \in \Lambda_\zeta}$
    \For{$n \in \{0, \dots, N\}$}
      \State $\Lambda_n = (\bsnu : \bsnu \in \Lambda_\zeta, \ d_\bsnu = n)$
      \State $\bstau = \left(\max (\set{\nu^s_i}{\bsnu \in \Lambda_n})\right)_{i=1}^n$
      \State $I^{\Lambda, n} = 0$
      \For{$\bsnu \in \Lambda_n$}
        \State $I^{\Lambda, n} += \zeta_{\Lambda, \bsnu} I^{\bsnu, s, \bstau}$
      \EndFor
    \State $I^\Lambda += I^{\Lambda, n}$
    \EndFor
    \State
    \Return $I^\Lambda$
  \end{algorithmic}
\end{algorithm}

While the previous discussion focused on scalar-valued interpolation targets (i.e., the case $d_2 = 1$), the extension to vector-valued functions is straightforward and works seamlessly, provided that all interpolants in the codomain are constructed using the same multi-index set $\Lambda$.

# Acknowledgements


# References
