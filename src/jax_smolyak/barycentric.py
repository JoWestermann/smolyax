import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def compute_weights(nodes: ArrayLike) -> jax.Array:
    """
    Compute the barycentric interpolation weights corresponding to given nodes.

    Parameters
    ----------
    nodes : ArrayLike
        A 1D array of shape (n,) containing n one-dimensional interpolation nodes.

    Returns
    -------
    ArrayLike
        The corresponding barycentric interpolation weights.
    """
    nodes = jnp.asarray(nodes)
    diffs = nodes[:, None] - nodes
    diffs = jnp.where(diffs == 0, 1, diffs)
    return jnp.prod(1 / diffs, axis=0)
