"""
Utility functionality for tensor product quadrature.
"""

from typing import Sequence

import jax
import jax.numpy as jnp


def evaluate_tensor_product_quadrature(F: jax.Array, w_list: Sequence[jax.Array]) -> jax.Array:
    """
    Evaluate the tensor product quadrature rule.

    Parameters
    ----------
    F : jax.Array
        Tensors storing the evaluations of the target function `f`.
        Should be a multi-dimensional array with shape `(d_out, mu_1, mu_2, ..., mu_n)`
        where each `mu_i` corresponds to the number of points in the ith dimension.

    w_list : Sequence[jax.Array]
        Quadrature weights. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

    Returns
    -------
    jax.Array
        The integral of the tensor product interpolant at the points specified by `x`.
        The shape of the output will be `(d_out,)`.
    """
    for w in w_list:
        F = jnp.einsum("j,kj...->k...", w, F)
    return F
