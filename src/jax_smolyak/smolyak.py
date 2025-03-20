import itertools as it
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from . import indices, nodes
from .tensorproduct import TensorProductBarycentricInterpolator


class SmolyakBarycentricInterpolator:

    @property
    def is_nested(self) -> bool:
        return self._is_nested

    def __init__(
        self, node_gen: nodes.Generator, k: ArrayLike, t: float, f: Callable = None
    ):
        """
        node_gen : interpolation node generator object
        k : weight vector of the anisotropy of the multi-index set (TODO: move construction of multi-index outside)
        t : threshold controlling the size of the multi-index set
        f : interpolation target function
        """
        self.k = k
        self.operators = []
        self.coefficients = []
        self._is_nested = node_gen.is_nested

        def kmap(j):
            return k[j]

        i = indices.indexset_sparse(kmap, t, cutoff=len(k))
        for nu in i:
            c = indices.smolyak_coefficient_zeta_sparse(kmap, t, nu=nu, cutoff=len(k))
            if c != 0:
                self.operators.append(
                    TensorProductBarycentricInterpolator(node_gen, nu, len(k))
                )
                self.coefficients.append(c)
        if self.is_nested:
            self.n = len(i)
        else:
            self.n = int(np.sum([np.prod(o.F.shape) for o in self.operators]))
        self.n_f_evals = 0
        if f is not None:
            self.set_f(f)

    def set_f(self, f: Callable, f_evals: dict = None, i: int = None):
        """
        Compute (or reuse pre-computed) evaluations of the target function f at the interpolation nodes.
        f : interpolation target function
        f_evals : dictionary mapping interpolation nodes to function evaluations
        i : index to access a specific dimension of the output (e.g. as used by
            MultivariateSmolyakBarycentricInterpolator)
        """
        if f_evals is None:
            f_evals = {}
        if self.is_nested:
            for o in self.operators:
                for idx in it.product(*(range(d + 1) for d in o.degrees)):
                    ridx = o.reduced_index(idx)
                    if idx not in f_evals.keys():
                        o.set_x(ridx)
                        f_evals[idx] = f(o.x)
                        self.n_f_evals += 1
                    if i is None:
                        o.F[ridx] = f_evals[idx]
                    else:
                        o.F[ridx] = f_evals[idx][i]
        else:
            for o in self.operators:
                f_evals_o = f_evals.get(o.degrees, {})
                for idx in it.product(*(range(d + 1) for d in o.degrees)):
                    ridx = o.reduced_index(idx)
                    if idx not in f_evals_o.keys():
                        o.set_x(ridx)
                        f_evals_o[idx] = f(o.x)
                        self.n_f_evals += 1
                    if i is None:
                        o.F[ridx] = f_evals_o[idx]
                    else:
                        o.F[ridx] = f_evals_o[idx][i]
                f_evals[o.degrees] = f_evals_o
        return f_evals

    def __call__(self, x: ArrayLike) -> ArrayLike:
        r = 0
        for c, o in zip(self.coefficients, self.operators):
            r += c * o(x)
        return r


class MultivariateSmolyakBarycentricInterpolator:

    def __init__(
        self,
        *,
        node_gen: nodes.Generator,
        k: ArrayLike,
        t: ArrayLike,
        f: Callable = None,
    ):
        """
        node_gen : interpolation node generator object
        k : weight vector of the anisotropy of the multi-index set (TODO: move construction of multi-index outside)
        t : threshold controlling the size of the multi-index set
        f : interpolation target function
        """
        self.components = [SmolyakBarycentricInterpolator(node_gen, k, ti) for ti in t]
        self.n = max(c.n for c in self.components)
        self.F = None
        if f is not None:
            self.set_f(f=f)

    def set_f(self, *, f: Callable, f_evals=None):
        """
        Compute (or reuse pre-computed) evaluations of the target function f at the interpolation nodes.
        f : interpolation target function
        f_evals : dictionary mapping interpolation nodes to function evaluations
        """
        assert self.F is None
        if f_evals is None:
            f_evals = {}
        for i, c in enumerate(self.components):
            f_evals = c.set_f(f, f_evals, i)
        self.F = f_evals

        return f_evals

    def __call__(self, x: ArrayLike) -> ArrayLike:
        res = np.array([c(x) for c in self.components]).T
        assert res.shape[res.ndim - 1] == len(self.components)
        return res
