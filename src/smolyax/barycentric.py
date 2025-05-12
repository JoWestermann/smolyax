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
