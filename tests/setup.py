from typing import List, Sequence

import numpy as np
from numpy.polynomial import hermite, legendre
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

        domain = np.sort(np.random.rand(d, 2), axis=1)
        sets.append(nodes.Leja(domains=domain))

        mean = np.random.randn(d)
        scaling = np.random.rand(d)
        sets.append(nodes.GaussHermite(mean, scaling))

    return sets


def evaluate_univariate_polynomial(node_gen_uni: nodes.Generator1D, degree: int, x: np.ndarray) -> ArrayLike:
    x = node_gen_uni.scale_back(x)
    coefficients = [0] * degree + [1]
    if isinstance(node_gen_uni, nodes.Leja1D):
        return legendre.legval(x, coefficients)
    elif isinstance(node_gen_uni, nodes.GaussHermite1D):
        return hermite.hermval(x, coefficients)
    else:
        raise TypeError(f"Unsupported node generator type: {type(node_gen_uni)}")


def differentiate_univariate_polynomial(node_gen_uni: nodes.Generator1D, degree: int, x: np.ndarray) -> ArrayLike:
    x = node_gen_uni.scale_back(x)
    coefficients = [0] * degree + [1]
    if isinstance(node_gen_uni, nodes.Leja1D):
        derivative_coefficients = legendre.legder(coefficients)
        scaling = 1 if node_gen_uni.domain is None else (node_gen_uni.domain[1] - node_gen_uni.domain[0]) / 2
        return legendre.legval(x, derivative_coefficients) / scaling
    elif isinstance(node_gen_uni, nodes.GaussHermite1D):
        derivative_coefficients = hermite.hermder(coefficients) / node_gen_uni.scaling
        return hermite.hermval(x, derivative_coefficients)
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
        print("\tTest polynomials with degrees", self.selected_idxs)

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

    def integral(self) -> ArrayLike:
        return np.array([np.prod([1 if n == 0 else 0 for g, n in zip(self.node_gen, nu)]) for nu in self.selected_idxs])

    def gradient(self, x: np.ndarray) -> ArrayLike:
        return np.array([self.__differentiate_multivariate_polynomial(nu, x) for nu in self.selected_idxs]).transpose(
            2, 0, 1
        )

    def __differentiate_multivariate_polynomial(self, nu: Sequence[int], x: np.ndarray) -> ArrayLike:
        if x.ndim == 1:
            p_x = [evaluate_univariate_polynomial(g, n, xi) for g, n, xi in zip(self.node_gen, nu, x)]
            dp_x = [differentiate_univariate_polynomial(g, n, xi) for g, n, xi in zip(self.node_gen, nu, x)]
            return [dp_x[j] * np.prod([p_x[i] for i in range(len(p_x)) if i != j]) for j in range(len(p_x))]
        elif x.ndim == 2:
            p_x = [evaluate_univariate_polynomial(g, n, x[:, i]) for i, (g, n) in enumerate(zip(self.node_gen, nu))]
            dp_x = [
                differentiate_univariate_polynomial(g, n, x[:, i]) for i, (g, n) in enumerate(zip(self.node_gen, nu))
            ]

            p_x = np.stack(p_x, axis=0)  # shape (d, n_samples)
            dp_x = np.stack(dp_x, axis=0)  # shape (d, n_samples)

            prod_px = np.prod(p_x, axis=0)  # shape (n_samples,)

            return np.where(p_x != 0, dp_x * prod_px / p_x, 0.0)  # gradient: shape (d, n_samples)
        else:
            raise ValueError("Input x must be 1D or 2D")
