r"""
# Overview

---

ℹ️ **For installation guidelines and examples see the
    [project README](https://github.com/JoWestermann/smolyax#readme).**

---

### `smolyax.interpolation`: Smolyak barycentric operator

Central class **`SmolyakBarycentricInterpolator`** constructs the interpolant to a multivariate and vector-valued
function $f$
$$
    I^{\Lambda_{\boldsymbol{k},t}} [f] =
    \sum \limits_{\boldsymbol{\nu} \in \Lambda_{\boldsymbol{k},t}}
    \zeta_{\Lambda_{\boldsymbol{k},t}, \boldsymbol{\nu}} I^\boldsymbol{\nu}[f]
$$
on sparse grids governed by anisotropic multi-index sets $\Lambda_{\boldsymbol{k},t}$.
Additionally supports computing the integral (which corresponds to a quadrature approximation of the integral of $f$),
as well as evaluating its gradient.

### `smolyax.indices`: Multi-index sets
Provides numba-accelerated routines to
* construct anisotropic total degree multi-index sets of the form
$$
    \Lambda_{\boldsymbol{k}, t} := \\{\boldsymbol{\nu} \in \mathbb{N}_0^{d} \ : \ \sum_{j=1}^{d} k_j \nu_j < t\\}.
$$
* compute the Smolyak coefficients $\zeta_{\Lambda_{\boldsymbol{k},t}, \boldsymbol{\nu}} := \sum \limits_{\boldsymbol{e}
    \in \\{0,1\\}^d : \boldsymbol{\nu}+\boldsymbol{e} \in \Lambda_{\boldsymbol{k},t}} (-1)^{|\boldsymbol{e}|}$.
* find thresholds $t$ that for given $\boldsymbol{k}$ allows to construct $\Lambda_{\boldsymbol{k}, t}$ with a
    specified target cardinality.

### `smolyax.nodes`: Interpolation node generators

Unifies different 1-D rules under an abstract `Generator1D` interface:

| Class | Domain | Nested |
|-------|--------|--------|
| `Leja1D` | bounded interval | yes |
| `GaussHermite1D` | $\mathbb R$ (Gaussian) | no |

The interface allows for affine scaling of the domains and random sampling.

A `Generator` container bundles per-dimension generators.

### `smolyax.barycentric` and `smolyax.quadrature`: Tensor-product kernels
Implements the barycentric formula in one dimension and extends it to high-dimensional tensor-product form
$I^\boldsymbol{\nu}$. Provides the accompanying expressions for integrals and gradients of the tensor-product terms.
The routines are JIT-compatible and serve as internal utilities for the high-level interpolator.
"""
