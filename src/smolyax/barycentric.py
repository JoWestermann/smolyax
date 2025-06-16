"""
Utility functionality for barycentric interpolation.
"""

from typing import Sequence, Union

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


def evaluate_basis_unnormalized(x: jax.Array, xi: jax.Array, w: jax.Array, nu_i: int) -> jax.Array:
    r"""
    Evaluate the barycentric basis numerator terms at given evaluation points.

    Computes $w_j / (x - \xi_j)$ for indices $j \le \nu_i$; other entries are masked to zero.
    If an evaluation point coincides with a node $\xi_j$ for $j \le \nu_i$, returns a one-hot indicator pattern
    restricted to those entries.

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
        Barycentric numerator terms for indices $j \le \nu_i$, or a one-hot indicator pattern if `x` coincides with a
        node. Shape `(n_points, m_i)`.
    """
    diffs = x - xi
    mask_cols = jnp.arange(diffs.shape[1]) <= nu_i
    mask_zero = jnp.any((diffs == 0) & mask_cols, axis=1)
    one_hot_pattern = (diffs == 0).astype(diffs.dtype)
    w_div_diffs = jnp.divide(w, diffs)
    b = jnp.where(mask_zero[:, None], one_hot_pattern, w_div_diffs)
    return jnp.where(mask_cols[None, :], b, 0)


def evaluate_tensor_product_interpolant(
    x: jax.Array,
    F: jax.Array,
    xi_list: Sequence[jax.Array],
    w_list: Sequence[jax.Array],
    sorted_dims: Sequence[int],
    sorted_degs: Sequence[int],
) -> jax.Array:
    """
    Evaluate the tensor product interpolant using the barycentric formulation.

    Parameters
    ----------
    x : jax.Array
        Points at which to evaluate the tensor product interpolant of the target function `f`.
        Should be a 2D array of shape `(n_points, d_in)` where `n_points` is the number of evaluation points
        and `d_in` is the dimension of the input domain.

    F : jax.Array
        Tensors storing the evaluations of the target function `f`.
        Should be a multi-dimensional array with shape `(d_out, mu_1, mu_2, ..., mu_n)`
        where each `mu_i` corresponds to the number of points in the ith dimension.

    xi_list : Sequence[jax.Array]
        Interpolation nodes. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

    w_list : Sequence[jax.Array]
        Interpolation weights. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

    sorted_dims : Sequence[int]
        Dimensions with nonzero interpolation degree.

    sorted_degs : Sequence[int]
        Interpolation degrees per dimension.

    Returns
    -------
    jax.Array
        The evaluated tensor product interpolant at the points specified by `x`.
        The shape of the output will be `(n_points, d_out)`.
    """
    norm = jnp.ones(x.shape[0])
    for i, (si, nui) in enumerate(zip(sorted_dims, sorted_degs)):
        b = evaluate_basis_unnormalized(x[:, [si]], xi_list[i], w_list[i], nui)
        if i == 0:
            F = jnp.einsum("ij,kj...->ik...", b, F)
        else:
            F = jnp.einsum("ij,ikj...->ik...", b, F)
        norm *= jnp.sum(b, axis=1)
    return F / norm[:, None]


def evaluate_basis_gradient_unnormalized(x: jax.Array, xi: jax.Array, w: jax.Array, nu_i: int) -> jax.Array:
    r"""
    Evaluate the gradient of the barycentric basis numerator terms at given evaluation points.

    Computes $-w_j / (x - \xi_j)^2$ for indices $j \le \nu_i$; other entries are masked to zero.
    If an evaluation point coincides with a node $\xi_j$ for $j \le \nu_i$, returns `NaN` at that position.

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
    jax.Array
        Gradient values for indices $j \le \nu_i$, or `NaN` where `x` coincides with a node. Shape: `(n_points, m_i)`
    """
    diffs = (x - xi) ** 2
    mask_cols = jnp.arange(diffs.shape[1]) <= nu_i
    singular_mask = (diffs == 0) & mask_cols
    w_div_diffs = jnp.divide(-w, diffs)
    w_div_diffs = jnp.where(singular_mask, jnp.nan, w_div_diffs)
    return jnp.where(mask_cols[None, :], w_div_diffs, 0)


def evaluate_tensor_product_gradient(
    x: jax.Array,
    F: jax.Array,
    xi_list: Sequence[jax.Array],
    w_list: Sequence[jax.Array],
    sorted_dims: Sequence[int],
    sorted_degs: Sequence[int],
) -> jax.Array:
    """
    Evaluate the gradient of the tensor product interpolant using the barycentric formulation.

    Parameters
    ----------
    x : jax.Array
        Points at which to evaluate the tensor product interpolant of the target function `f`.
        Should be a 2D array of shape `(n_points, d_in)` where `n_points` is the number of evaluation points
        and `d_in` is the dimension of the input domain.

    F : jax.Array
        Tensors storing the evaluations of the target function `f`.
        Should be a multi-dimensional array with shape `(d_out, mu_1, mu_2, ..., mu_n)`
        where each `mu_i` corresponds to the number of points in the ith dimension.

    xi_list : Sequence[jax.Array]
        Interpolation nodes. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

    w_list : Sequence[jax.Array]
        Interpolation weights. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

    sorted_dims : Sequence[int]
        Dimensions with nonzero interpolation degree.

    sorted_degs : Sequence[int]
        Interpolation degrees per dimension.

    Returns
    -------
    jax.Array
        The gradient of the tensor product interpolant evaluated at the points specified by `x`.
        The shape of the output will be `(n_points, d_out, d_in)`.
    """
    n = len(sorted_dims)
    d_out = F.shape[0]
    d_in = x.shape[1]
    n_points = x.shape[0]

    for i, (si, nui) in enumerate(zip(sorted_dims, sorted_degs)):
        b = evaluate_basis_unnormalized(x[:, [si]], xi_list[i], w_list[i], nui)
        b_norm = jnp.sum(b, axis=1)[:, None]

        db = evaluate_basis_gradient_unnormalized(x[:, [si]], xi_list[i], w_list[i], nui)
        db_norm = jnp.sum(db, axis=1)[:, None]

        b = b / b_norm
        B_i = db / b_norm - b * db_norm / b_norm
        B = jnp.tile(b[:, None, :], (1, n, 1))
        B = B.at[:, i, :].set(B_i)

        if i == 0:
            F = jnp.einsum("ihj,kj...->ikh...", B, F)
        else:
            F = jnp.einsum("ihj,ikhj...->ikh...", B, F)

    result = jnp.zeros((n_points, d_out, d_in))
    for i, si in enumerate(sorted_dims):
        result = result.at[:, :, si].set(F[:, :, i])
    return result
