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
            self.set_F(f)

    def set_F(self, f: Callable, F: dict = None, i: int = None):
        if F is None:
            F = {}
        if self.is_nested:
            for o in self.operators:
                for idx in it.product(*(range(d + 1) for d in o.degrees)):
                    ridx = o.reduced_index(idx)
                    if idx not in F.keys():
                        o.set_x(ridx)
                        F[idx] = f(o.x)
                        self.n_f_evals += 1
                    if i is None:
                        o.F[ridx] = F[idx]
                    else:
                        o.F[ridx] = F[idx][i]
        else:
            for o in self.operators:
                Fo = F.get(o.degrees, {})
                for idx in it.product(*(range(d + 1) for d in o.degrees)):
                    ridx = o.reduced_index(idx)
                    if idx not in Fo.keys():
                        o.set_x(ridx)
                        Fo[idx] = f(o.x)
                        self.n_f_evals += 1
                    if i is None:
                        o.F[ridx] = Fo[idx]
                    else:
                        o.F[ridx] = Fo[idx][i]
                F[o.degrees] = Fo
        return F

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
        self.components = [SmolyakBarycentricInterpolator(node_gen, k, ti) for ti in t]
        self.n = max(c.n for c in self.components)
        self.F = None
        if f is not None:
            self.set_F(f=f)

    def set_F(self, *, f: Callable, F=None):
        assert self.F is None
        if F is None:
            F = {}
        for i, c in enumerate(self.components):
            F = c.set_F(f, F, i)
        self.F = F

        return F

    def __call__(self, x: ArrayLike) -> ArrayLike:
        res = np.array([c(x) for c in self.components]).T
        assert res.shape[res.ndim - 1] == len(self.components)
        return res
