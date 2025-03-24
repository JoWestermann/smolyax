from typing import Callable, List

import numpy as np
from numpy.polynomial.hermite import hermval
from numpy.polynomial.legendre import legval
from numpy.typing import ArrayLike

from jax_smolyak import indices, nodes


def generate_nodes_default(*, dmin: int, dmax: int) -> List[nodes.Generator]:
    sets = []
    for d in range(dmin, dmax + 1):
        sets.append(nodes.Leja(dim=d))
        sets.append(nodes.GaussHermite(dim=d))
    return sets


def generate_nodes(*, n: int, dmin: int, dmax: int) -> List[nodes.Generator]:
    sets = []
    for _ in range(n):

        d = np.random.randint(low=dmin, high=dmax + 1)

        domain = np.zeros((d, 2))
        domain[:, 1] = np.sort(np.random.rand(d) * 10)
        domain[:, 0] = -domain[:, 1]
        sets.append(nodes.Leja(domains=domain))

        mean = np.random.randn(d)
        scaling = np.random.rand(d)
        sets.append(nodes.GaussHermite(mean, scaling))

    return sets


def evaluate_univariate_polynomial(
    node_gen: nodes.Generator, nu: int, x: ArrayLike
) -> ArrayLike:
    x = node_gen.scale_back(x)
    if isinstance(node_gen, nodes.Leja):
        return legval(x, [0] * nu + [1])
    else:
        return hermval(x, [0] * nu + [1])


def evaluate_multivariate_polynomial(
    node_gen: nodes.Generator, nu: ArrayLike, x: ArrayLike
) -> ArrayLike:
    if x.ndim <= 1:
        return np.prod(
            [
                evaluate_univariate_polynomial(gi, nui, xi)
                for gi, nui, xi in zip(node_gen, nu, x)
            ]
        )
    elif x.ndim == 2:
        return np.prod(
            np.array(
                [
                    evaluate_univariate_polynomial(gi, nui, xi)
                    for gi, nui, xi in zip(node_gen, nu, x.T)
                ]
            ),
            axis=0,
        ).T
    else:
        raise


def generate_test_function_tensorproduct(
    *, node_gen: nodes.Generator, nu: ArrayLike
) -> Callable:
    if np.isscalar(nu):
        return lambda x: evaluate_univariate_polynomial(node_gen, nu, x)
    return lambda x: evaluate_multivariate_polynomial(node_gen, nu, x)


def generate_test_function_smolyak(
    *, node_gen: nodes.Generator, k: ArrayLike, t: ArrayLike, d_out: int
) -> Callable:
    if np.isscalar(t):
        t = [t] * d_out
    assert len(t) == d_out

    selected_idxs = []
    for ti in t:
        idxs = indices.indexset_sparse(k, ti, cutoff=len(k))
        j = np.random.randint(len(idxs))
        selected_idxs.append(indices.sparse_index_to_dense(idxs[j], cutoff=len(k)))
    print("\t Test polynomials with degrees", selected_idxs)
    return lambda x: np.array(
        [evaluate_multivariate_polynomial(node_gen, nu, x) for nu in selected_idxs]
    ).T
