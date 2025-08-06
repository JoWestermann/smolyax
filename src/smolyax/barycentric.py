"""
Utility functionality for barycentric interpolation.
"""

import string
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


def evaluate_basis_unnormalized_vectorized(x: jax.Array, xi: jax.Array, w: jax.Array, nu_i: int) -> jax.Array:
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
    diffs = x[:, :, None] - xi[:, None, :]

    mask_cols = jnp.arange(xi.shape[-1])[None, :] <= nu_i[:, None]
    mask_cols = mask_cols[:, None, :]

    mask_zero = jnp.any((diffs == 0) & mask_cols, axis=-1)
    one_hot = (diffs == 0).astype(diffs.dtype)
    w_div_diff = w[:, None, :] / diffs

    b = jnp.where(mask_zero[..., None], one_hot, w_div_diff)
    return jnp.where(mask_cols, b, 0)


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


@jax.jit
def evaluate_tensor_product_interpolant(
    x: jax.Array,
    F: jax.Array,
    xi_list: jax.Array,
    w_list: jax.Array,
    sorted_dims: Sequence[int],
    sorted_degs: Sequence[int],
    zeta: int,
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

    xi_list : jax.Array
        Interpolation nodes. A sequence of `n` stacked 1D arrays of length `max(mu_1, mu_2, ..., mu_n)`

    w_list : jax.Array
        Interpolation weights. A sequence of `n` stacked 1D arrays of length `max(mu_1, mu_2, ..., mu_n)`

    sorted_dims : Sequence[int]
        Dimensions with nonzero interpolation degree.

    sorted_degs : Sequence[int]
        Interpolation degrees per dimension.

    zeta : int
        Smolyak coefficient.

    Returns
    -------
    jax.Array
        The evaluated tensor product interpolant at the points specified by `x`.
        The shape of the output will be `(n_points, d_out)`.
    """
    x_sel = jnp.take(x, sorted_dims, axis=1).transpose(2, 1, 0)
    bs = []
    for i in range(sorted_degs.shape[1]):
        b = evaluate_basis_unnormalized_vectorized(x_sel[i], xi_list[i], w_list[i], sorted_degs[:, i])
        bs += [b / b.sum(axis=-1)[..., None]]

    letters = string.ascii_letters
    F_axes, in_axis, batch_axis = letters[: F.ndim - 1], letters[F.ndim - 1], "z"
    subscripts = (
        f"{batch_axis},{','.join([batch_axis+F_axes]+[batch_axis+in_axis+F_axis for F_axis in F_axes[1:]])}"
        + "->{in_axis}{F_axes[0]}"
    )

    return jnp.einsum(subscripts, zeta, F, *bs)


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


def evaluate_basis_gradient_unnormalized_vectorized(
    x: jax.Array,
    xi: jax.Array,
    w: jax.Array,
    nu_i: jax.Array,
) -> jax.Array:
    r"""
    Vectorised evaluation of the gradient of the barycentric basis numerator terms.

    Computes ``-w_j / (x - ξ_j)²`` for all indices ``j ≤ ν_i``; other entries are masked to zero.
    If an evaluation point coincides with a node ``ξ_j`` for an allowed index, the corresponding
    entry is set to ``NaN``.

    Parameters
    ----------
    x : jax.Array
        Evaluation points. Shape ``(batch, n_points, 1)``.
    xi : jax.Array
        Interpolation nodes. Shape ``(batch, m_i)``.
    w : jax.Array
        Barycentric weights for the nodes. Shape ``(batch, m_i)``.
    nu_i : jax.Array
        Polynomial degree per batch item. Shape ``(batch,)``.

    Returns
    -------
    jax.Array
        Gradient values with shape ``(batch, n_points, m_i)``.
    """
    # pair-wise differences and their squares
    diffs = x[:, :, None] - xi[:, None, :]  # (B, n, m_i)
    sq_diffs = diffs**2  # (B, n, m_i)

    # column mask: j ≤ ν_i  (broadcast batch-wise, then add singleton axis for n_points)
    mask_cols = (jnp.arange(xi.shape[-1])[None, :] <= nu_i[:, None])[:, None, :]  # (B, 1, m_i)

    # singularities: x == ξ_j for an admissible column
    singular = (sq_diffs == 0) & mask_cols  # (B, n, m_i)

    # gradient −w / (x − ξ)²  with NaN at singularities
    grad = -w[:, None, :] / sq_diffs  # (B, n, m_i)
    grad = jnp.where(singular, jnp.nan, grad)

    # zero-out columns j > ν_i
    return jnp.where(mask_cols, grad, 0)


def assemble_B(i, n, x, nui, ni, wi):
    b = evaluate_basis_unnormalized_vectorized(x, ni, wi, nui)
    b_norm = b.sum(axis=-1)[..., None]

    db = evaluate_basis_gradient_unnormalized_vectorized(x, ni, wi, nui)
    db_norm = db.sum(axis=-1)[..., None]

    b = b / b_norm
    B_i = db / b_norm - b * db_norm / b_norm
    B = jnp.tile(b[:, :, None, :], (1, 1, n, 1))
    B = B.at[:, :, i, :].set(B_i)
    return B


def evaluate_tensor_product_gradient(
    x: jax.Array,
    F: jax.Array,
    xi_list: jax.Array,
    w_list: jax.Array,
    sorted_dims: Sequence[int],
    sorted_degs: Sequence[int],
    zeta: int,
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

    xi_list : jax.Array
        Interpolation nodes. A sequence of `n` stacked 1D arrays of length `max(mu_1, mu_2, ..., mu_n)`

    w_list : jax.Array
        Interpolation weights. A sequence of `n` stacked 1D arrays of length `max(mu_1, mu_2, ..., mu_n)`

    sorted_dims : Sequence[int]
        Dimensions with nonzero interpolation degree.

    sorted_degs : Sequence[int]
        Interpolation degrees per dimension.

    zeta : int
        Smolyak coefficient.

    Returns
    -------
    jax.Array
        The gradient of the tensor product interpolant evaluated at the points specified by `x`.
        The shape of the output will be `(n_points, d_out, d_in)`.
    """
    output_shape = (F.shape[0], x.shape[0], F.shape[1], x.shape[1])
    n = F.ndim - 2

    x_sel = jnp.take(x, sorted_dims, axis=1).transpose(2, 1, 0)
    Bs = []
    for i in range(sorted_degs.shape[1]):
        Bs += [assemble_B(i, n, x_sel[i], sorted_degs[:, i], xi_list[i], w_list[i])]
    # B = assemble_B(0, n, x, sorted_dims[0], sorted_degs[0], xi_list[0], w_list[0])
    # F = jnp.einsum("ihj,kj...->ikh...", B, F)

    # for i, (si, nui, ni, wi) in enumerate(zip(sorted_dims[1:], sorted_degs[1:], xi_list[1:], w_list[1:])):
    #    B = assemble_B(i+1, n, x, si, nui, ni, wi)
    #    F = jnp.einsum("ihj,ikhj...->ikh...", B, F)

    dims = string.ascii_letters[3 : n + 3]
    subscripts = ",".join([f"Zb{''.join(dims)}"] + [f"Zac{j}" for j in dims]) + "->Zabc"

    F = jnp.einsum(subscripts, F, *Bs)
    print(output_shape, sorted_dims.shape, F.shape)

    n_data, n_pts, k1, k2 = F.shape

    row_idx = jnp.arange(n_data)[:, None, None]  # (n_data,1,1)
    col_idx = jnp.arange(n_pts)[None, :, None]  # (1,n_pts,1)
    dim_idx = jnp.broadcast_to(sorted_dims[:, None, :], (n_data, n_pts, k1))

    result = (
        jnp.zeros(output_shape, dtype=F.dtype)  # (n_data,n_pts,m_max,k2)
        .at[row_idx, col_idx, dim_idx]  # indices along axes 0,1,2
        .set(F)  # update tensor (n_data,n_pts,k1,k2)
    )

    # result = jnp.zeros(output_shape).at[:, :, :, sorted_dims].set(F)
    print(result.shape, zeta.shape)
    return jnp.einsum("i...,i...->...", zeta, result)
