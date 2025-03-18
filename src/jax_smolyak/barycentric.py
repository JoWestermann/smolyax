import numpy as np
from numpy.typing import ArrayLike


def compute_weights(nodes: ArrayLike) -> ArrayLike:
    """
    nodes: np.array of shape (n, ) containing n one-dimensional interpolation nodes
    returns: corresponding barycentric interpolation weights
    """
    diffs = nodes[:, None] - nodes
    np.fill_diagonal(diffs, 1)
    return np.prod(1 / diffs, axis=0)
