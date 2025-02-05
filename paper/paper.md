---
title: 'jax-smolyak: A HPC-capable implementation of the Smolyak interpolation operator'
tags:
  - Python
  - JAX
  - Interpolation
  - HPC
  - Smolyak
  - Sparse Grids
authors:
  - name: Josephine Westermann
    orcid: 0000-0000-0000-0000
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

**[TODO]**

# Statement of need

**[TODO]**

# High-dimensional interpolation with the Smolyak operator

We briefly summarize the essentials of high-dimensional interpolation, where sparse grids have become the standard choice for interpolation points. In this setting, the interpolation operator is commonly referred to as the Smolyak operator. This overview provides background and establishes notation, which will be used to describe our specific implementation choices in the next section.

**Univariate interpolation.** Given a domain $D \subset \R$ and set of $\ell \in \N$ pairwise distinct interpolation points $(\xi^\ell_i)_{i=0}^\ell \subset D$, the polynomial interpolation operator $I^\ell : C^0(D) \to \bbP_\ell := {\rm span} \set{x^i}{i=0,\dots,\ell}$ is defined as the mapping of a function $f$ onto the (unique) polynomial $I^\ell [f]$ of maximal degree $\ell$ such that $f(\xi^\ell_i) = I^\ell [f](\xi^\ell_i)$ for all $i\in\{0,1,\dots,\ell\}$.

One way to compute the interpolating polynomial that is known for its numerical stability is barycentric interpolation [@berrut:2004]. The univariate barycentric interpolation formula is given as
\begin{align}
    I^\ell [f] (x) := \frac{\sum_{i=0}^\ell b_i^\ell(x) f(\xi^\ell_i)}{\sum_{i=0}^\ell b_i^\ell(x)},
    \qquad \text{ with }
    b_i^\ell(x) &:= \begin{cases} 1 &\text{ if } i = \ell = 0 \\ \frac{w_i^\ell}{x-\xi^\ell_i} &\text{ else }\end{cases} \\
    \quad \text{ and }
    w_i^\ell &:= \prod \limits_{k\in\{0,1,...,\ell\}/\{i\}}\frac{1}{(\xi^\ell_i - \xi^\ell_k)}.
\end{align}

**Tensorized interpolation.** Let $d\in \N$. Given a tensor-product domain $D = \otimes_{j=1}^d D_1$ with $D_1 \subset \R$ and a multi-index $\bsnu \in \N_0^d$ characterizing the maximal polynomial degree in each dimension, we define $I^\bsnu : C^0(D) \to \bbP_\bsnu := {\rm span} \set{\bsx^\bsmu}{\bsmu \leq \bsnu}$ as
\begin{equation}
    I^\bsnu := \otimes_{j=1}^d I^{\nu_j}.
\end{equation}

In this setting, the tensorized barycentric interpolation formula can be expressed as a repeated vector-tensor product
\begin{equation} \label{eq:ip_tensorproduct}
    I^\bsnu [f] (\bsx) =
    \frac{\bsb^{\nu_1}(x_1) \bsb^{\nu_2}(x_2) \cdots \bsb^{\nu_d}(x_d) \bsF^\bsnu}
    {\prod_{j=1}^d (\sum_{i=0}^{\nu_j} b_i^{\nu_j} (x_j))},
\end{equation}
with tensors $\bsF^\bsnu \in \R^{\bsnu+{\bm 1}} := \R^{(\nu_1+1) \times (\nu_2+1) \times \dots \times (\nu_d+1)}$ given as
\begin{align*}
  F^\bsnu_{\bsmu} := f(\bsxi^\bsnu_\bsmu)  \text{ with }
   \bsxi_\bsmu := \left(\xi^{\nu_1}_{\mu_1}, \xi^{\nu_2}_{\mu_2}, \dots, \xi^{\nu_d}_{\mu_d} \right) \quad \text{ for all }
   \bsmu \in \N_0^d \text{ s.t. } \bsmu \le \bsnu,
\end{align*}
and vectors $\bsb^{\nu_j} (x_j) \in \R^{\nu_j+1}$ given as
\begin{equation*}
  \bsb^{\nu_j}(x_j) := (b_i^{\nu_j}(x_j))_{i=0}^{\nu_j} \quad \text{ for all }
  j \in \{1, \dots, d\}.
\end{equation*}

**Smolyak interpolation** [@smolyak:1963; @barthelmann:2000] overcomes the curse-of-dimensionality that characterizes tensor-product interpolation \eqref{eq:ip_tensorproduct}, by introducing polynomial ansatz spaces $\bbP_\Lambda := {\rm span} \set{\bsx^\bsmu}{\bsmu \in \Lambda}$ parametrized by downward closed multi-index sets $\Lambda \subset \N_0^d$. The resulting interpolation operator is then a linear combination of tensorized interpolation operators:
\begin{equation} \label{eq:ip_smolyak}
    I^\Lambda := \sum \limits_{\bsnu \in \Lambda} \zeta_{\Lambda, \bsnu} I^\bsnu, \qquad \zeta_{\Lambda, \bsnu} := \sum \limits_{\bse \in \{0,1\}^d : \bsnu+\bse \in \Lambda} (-1)^{|\bse|}.
\end{equation}

# Vectorizable implementation of the Smolyak operator for HPC

To leverage key HPC techniques such as vectorization, parallelization, and batch processing, input data must conform to a uniform structure. However, the vectors and tensors in \eqref{eq:ip_smolyak} together with \eqref{eq:ip_tensorproduct} exhibit a wide range of shapes, which poses a challenge for efficient processing. In this section, we outline how a combination of squeezing, permuting, and padding can be used to reorganize these tensors into a small number of large, structured data blocks, enabling efficient computation while only incurring a modest memory overhead.

**Squeezing and ordering the dimensions of the tensorized interpolator.**
For any multi-index $\bsnu \in \N_0^d$, denote with $t(\bsnu)$ the tuple consisting of the elements in $\{j \in \{1, ..., d\} \ : \ \nu_j > 0\}$ that order the non-zero entries of $\bsnu$ descendingly. Let further $\bsnu^t := (\nu_j)_{j \in t(\bsnu)}$ and $d_\bsnu := |\bsnu_t| \equiv |t(\bsnu)| = | \set{j \in \{1, ..., d\}}{\nu_j > 0}| \le d$.

_Example: For the multi-index $\bsnu = (3,0,2,0,4) \in \N_0^5$, it holds that $t(\bsnu) = (5,1,3)$, $\bsnu^t = (4,3,2)$ and $d_\bsnu = 3$._

With the above notation we can re-express the tensorized interpolation operator for $\bsnu$ as
\begin{equation} \label{eq:ip_truncating}
  I^\bsnu [f] (\bsx) \equiv I^{\bsnu, t} [f] (\bsx) :=
    \frac{\bsb^{\nu^t_1}(x_{t_1(\bsnu)}) \bsb^{\nu^t_2}(x_{t_2(\bsnu)}) \cdots \bsb^{\nu^t_{d_\bsnu}}(x_{t_{d_\bsnu}(\bsnu)}) \bsF^{\bsnu,t}}
    {\prod_{j=1}^{d_\bsnu} (\sum_{i=0}^{\nu^t_j} b_i^{\nu^t_j} (x_{t_j(\bsnu)}))}
    \quad \forall \bsx \in D,
\end{equation}
with $\bsF^{\bsnu,t} \in \R^{\bsnu^t + {\bm 1}}$ given as $F^{\bsnu,t}_{\bsmu} := f(\bsxi^{\bsnu, t}_\bsmu)$ for all $\bsmu \in \N_0^{d_\bsnu}$ s.t. $\bsmu \le \bsnu^t$, with
\begin{align*}
    \left(\bsxi^{\bsnu, t}_\bsmu\right)_j :=
    \begin{cases}
    \xi^{\nu_j}_{\mu_i} &\text{ if } j \in t(\bsnu) \text{ and } j=t_i(\bsnu)\\
    \xi^{\nu_j}_0 &\text{ else.}
    \end{cases}
\end{align*}
This construction is equivalent to \eqref{eq:ip_tensorproduct} since for any dimension $j$ with $\nu_j = 0$ it holds that $\bsb^{\nu_j}(x_j) = (1)_{i=0}^0$.

**Padding tensorized interpolators to a common shape.** For any multi-index $\bstau \in \N_0^k$, define the padding operator $p^\bstau$ that acts on any order-$k$ tensor $\bsT \in \R^{\bsrho}$ with $\bsrho \le \bstau$ as
$$p^\bstau(\bsT) \in \R^{\bstau}, \ p^\bstau(\bsT)_{\bsmu} :=
\begin{cases}
    T_{\bsmu} &\text{ if } \bsmu \le \bsrho\\
    0 &\text{ else.}
\end{cases} \quad \forall \bsmu \le \bstau.$$
Any tensorized interpolation operator $I^\bsnu$ can be padded to higher dimensions $\bstau \ge \bsnu$ via
\begin{equation} \label{eq:ip_padding}
    I^\bsnu [f] (\bsx) \equiv I^{\bsnu, \bstau}[f] (\bsx) :=
    \frac{p^{\tau_1}(\bsb^{\nu_1}(x_1)) p^{\tau_2}(\bsb^{\nu_2}(x_2)) \cdots p^{\tau_d}(\bsb^{\nu_d}(x_d)) p^\bstau(\bsF^\bsnu)}
    {\prod_{j=1}^d (\sum_{i=0}^{\tau_j} p^{\tau_j}_i(\bsb^{\nu_j} (x_j)))}.
\end{equation}

We write $I^{\bsnu, t, \bstau}$ when applying \eqref{eq:ip_truncating} and \eqref{eq:ip_padding} successively.

**Pseudocode of our implementation strategy. [WIP]**
We now have all the ingredients in place to implent the Smolyak operator in a highly vectorizable way. Algorithm 1 summarizes the essential steps. Given a downward closed (but otherwise arbitrarily structured) multi-index set $\Lambda$ and the interpolation target $f$, we start with the trivial steps of determining the set $\Lambda_\zeta$ of multi-indices $\bsnu$ with non-zero Smolyak coefficient $\zeta_{\Lambda, \bsnu}$ and the largest number of non-zero multi-index entries $K$. This number typically remains small (single-digit), even when the number of dimensions $d$ is in the hundreds and the cardinality of the multi-index set is in the ten thousands. Now, for each fixed number $k$ of non-zero entries, we determine the set $\Lambda_k$ of all multi-indices with $k$ non-zero entries. For this subset, we determine the

\begin{algorithm}[H]
  \caption{Construct the multivariate barycentric Smolyak interpolator $I^\Lambda$\\
    \textit{Input:} Target function $f$, multi-index set $\Lambda \subset \N_0^d$\\
    \textit{Output:} $I^\Lambda$}
  \begin{algorithmic}[1]
    \State $I^\Lambda = 0$
    \State $\Lambda_\zeta = \set{\bsnu \in \Lambda}{\zeta_{\Lambda, \bsnu} \neq 0}$
    \State $K = \max \set{d_\bsnu}{\bsnu \in \Lambda_\zeta}$
    \For{$k \in \{1, \dots, K\}$}
      \State $\Lambda_k = (\bsnu : \bsnu \in \Lambda_\zeta, \ d_\bsnu = k)$
      \State $\bstau = \max (\set{\bsnu^t}{\bsnu \in \Lambda_k})$
      \State $I^{\Lambda, k} = 0$
      \For{$\bsnu \in \Lambda_k$}
        \State $I^{\Lambda, k} += \zeta_{\Lambda, \bsnu} I^{\bsnu, t, \tau}$
      \EndFor
    \State $I^\Lambda += I^{\Lambda, k}$
    \EndFor
    \State
    \Return $I^\Lambda$
  \end{algorithmic}
\end{algorithm}

# Scope of jax-smolyak [WIP]

The library is able to interpolate any multivariate and vector-valued function $f : \mathbb{R}^{d_1} \to \mathbb{R}^{d_2}$ for any $d_1, d_2 \in \N$. Note that while the above description of the method considered only scalar-valued interpolation targets (i.e. the case $d_2 = 1$), the step towards vector-values targets is trivial as long as all dimensions in the co-domain should be constructed using the same multi-index. But even dimension-specific ansatz spaces are possible with our implementation by stacking multiple interpolators.

The quality of the interpolation will depend on the smoothness of $f$ and the choice of interpolation nodes $\xi$ and multi-index sets $\Lambda$. **[TODO references]**

This repository is shipped with ...

Further interpolation nodes or multi-index sets can be added easily if necessary, by implementing a minimalistic interface.


# Acknowledgements


# References
