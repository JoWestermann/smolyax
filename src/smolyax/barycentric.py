"""
Utility functionality for barycentric interpolation.
"""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def compute_weights(nodes: Union[jax.Array, np.ndarray]) -> jax.Array:
    """
    Compute the barycentric interpolation weights corresponding to given nodes.

    Parameters
    ----------
    nodes : Union[jax.Array, np.ndarray]
        A 1D array of shape (n,) containing n one-dimensional interpolation nodes.

    Returns
    -------
    jax.Array
        The corresponding barycentric interpolation weights.
    """
    nodes = jnp.asarray(nodes)
    diffs = nodes[:, None] - nodes
    diffs = jnp.where(diffs == 0, 1, diffs)
    return jnp.prod(1 / diffs, axis=0)


def evaluate_basis_numerator_centered(x: jax.Array, xi: jax.Array, w: jax.Array, nu_i: int):
    r"""
    Evaluate the barycentric basis numerator terms at given evaluation points (assuming a centered domain).

    Computes the vector of terms $w_j / (x - \xi_j)$ for a single univariate interpolant.
    If an evaluation point coincides with a node, returns a one-hot indicator pattern.
    All entries of xi and w beyond index `nu_i` are assumed to be zero (which is not checked for efficiency).

    Parameters
    ----------
    x : jax.Array
        Evaluation points along one dimension. Shape: `(n_points, 1)`
    xi : jax.Array
        Interpolation nodes. Shape `(m_i,)` with `m_i > nu_i`. Entries beyond `nu_i` are assumed to be zero.
    w : jax.Array
        Barycentric weights corresponding to the given interpolation nodes. Shape `(m_i,)` with `m_i > nu_i`.
        Entries beyond `nu_i` are assumed to be zero.
    nu_i : int
        Polynomial degree in this dimension (unused here, included for interface compatibility).

    Returns
    -------
    b : jax.Array
        Barycentric numerator terms $w_j / (x - x_j)$, or a one-hot indicator pattern if `x` coincides with a node.
        Shape `(n_points, m_i)`.
    """
    diffs = x - xi
    mask_zero = jnp.any(diffs == 0, axis=1)
    one_hot_pattern = jnp.where(diffs == 0, 1.0, 0.0)
    w_div_diffs = w / diffs
    return jnp.where(mask_zero[:, None], one_hot_pattern, w_div_diffs)


def evaluate_basis_numerator_noncentered(x: jax.Array, xi: jax.Array, w: jax.Array, nu_i: int):
    r"""
    Evaluate the barycentric basis numerator terms at given evaluation points (assuming a non-centered domain).

    Computes $w_j / (x - \xi_j)$ for indices $j ≤ \nu_i$; other entries are masked to zero.
    If an evaluation point coincides with a node $\xi_j$ for $j ≤ \nu_i$, returns a one-hot indicator pattern restricted
    to those entries.

    Parameters
    ----------
    x : jax.Array
        Evaluation points along one dimension. Shape: `(n_points, 1)`
    xi : jax.Array
        Interpolation nodes. Shape `(m_i,)` with `m_i > nu_i`. Entries beyond `nu_i` are not used.
    w : jax.Array
        Barycentric weights corresponding to the given interpolation nodes. Shape `(m_i,)` with `m_i > nu_i`.
        Entries beyond `nu_i` are not used.
    nu_i : int
        Polynomial degree in this dimension.

    Returns
    -------
    b : jax.Array
        Barycentric numerator terms for indices $j ≤ \nu_i$, or a one-hot indicator pattern if `x` coincides with a
        node. Shape `(n_points, m_i)`.
    """
    diffs = x - xi
    mask_cols = jnp.arange(diffs.shape[1]) <= nu_i
    mask_zero = jnp.any((diffs == 0) & mask_cols, axis=1)
    one_hot_pattern = (diffs == 0).astype(diffs.dtype)
    w_div_diffs = jnp.divide(w, diffs)
    b = jnp.where(mask_zero[:, None], one_hot_pattern, w_div_diffs)
    return jnp.where(mask_cols[None, :], b, 0)
