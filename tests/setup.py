from typing import List, Sequence

import numpy as np
from numpy.polynomial.hermite import hermval
from numpy.polynomial.legendre import legval
from numpy.typing import ArrayLike

from smolyax import indices, nodes


def sparse_index_to_dense(nu: tuple[tuple[int, int], ...], dim: int) -> tuple:
    dense_nu = [0] * dim
    for k, v in nu:
        dense_nu[k] = v
    return tuple(dense_nu)


def dense_index_to_sparse(dense_nu: tuple[int]) -> tuple[tuple[int, int], ...]:
    sparse_nu = []
    for k, v in enumerate(dense_nu):
        if v > 0:
            sparse_nu.append((k, v))
    return tuple(sparse_nu)


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


def evaluate_univariate_polynomial(node_gen_uni: nodes.Generator, degree: int, x: np.ndarray) -> ArrayLike:
    x = node_gen_uni.scale_back(x)
    coefficients = [0] * degree + [1]
    if isinstance(node_gen_uni, nodes.Leja1D):
        return legval(x, coefficients)
    elif isinstance(node_gen_uni, nodes.GaussHermite1D):
        return hermval(x, coefficients)
    else:
        raise TypeError(f"Unsupported node generator type: {type(node_gen_uni)}")


class TestPolynomial:

    def __init__(self, *, node_gen: nodes.Generator, k: ArrayLike, t: ArrayLike, d_out: int):
        self.node_gen = node_gen

        if np.isscalar(t):
            t = [t] * d_out
        assert len(t) == d_out

        self.selected_idxs = []
        for ti in t:
            idxs = indices.indexset(k, ti)
            j = np.random.randint(len(idxs))
            self.selected_idxs.append(sparse_index_to_dense(idxs[j], dim=len(k)))
        print("\t Test polynomials with degrees", self.selected_idxs)

    def __call__(self, x: np.ndarray) -> ArrayLike:
        return np.array([self.__evaluate_multivariate_polynomial(nu, x) for nu in self.selected_idxs]).T

    def __evaluate_multivariate_polynomial(self, nu: Sequence[int], x: np.ndarray) -> ArrayLike:
        if x.ndim == 1:
            return np.prod([evaluate_univariate_polynomial(g, n, xi) for g, n, xi in zip(self.node_gen, nu, x)])
        elif x.ndim == 2:
            return np.prod(
                [evaluate_univariate_polynomial(g, n, x[:, i]) for i, (g, n) in enumerate(zip(self.node_gen, nu))],
                axis=0,
            )
        else:
            raise ValueError("Input x must be 1D or 2D")
