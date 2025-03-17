import copy
import itertools as it
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from . import indices

jax.config.update("jax_enable_x64", True)


def __evaluate_tensorproduct_interpolant(
    x: ArrayLike, F: ArrayLike, xi_list: List, w_list: List, s_list: List
) -> ArrayLike:
    norm = jnp.ones(x.shape[0])
    for i, si in enumerate(s_list):
        b = x[:, [si]] - xi_list[i]
        has_zero = jnp.any(b == 0, axis=1)
        zero_pattern = jnp.where(b == 0, 1.0, 0.0)
        normal = w_list[i] / b
        b = jnp.where(has_zero[:, None], zero_pattern, normal)
        if i == 0:
            F = jnp.einsum("ij,kj...->ik...", b, F)
        else:
            F = jnp.einsum("ij,ikj...->ik...", b, F)
        norm *= jnp.sum(b, axis=1)
    return F / norm[:, None]


def _create_evaluate_tensorproduct_interpolant_for_vmap(n: int):
    def wrapped_function(x, F, *args):
        xi_list = args[:n]
        w_list = args[n : 2 * n]
        s_list = args[2 * n]
        return __evaluate_tensorproduct_interpolant(x, F, xi_list, w_list, s_list)

    return jax.jit(wrapped_function)


class MultivariateSmolyakBarycentricInterpolator:

    @property
    def is_nested(self) -> bool:
        return self._is_nested

    def __init__(
        self, *, g, k, t, d_out: int, f: Callable = None, batchsize: int = 250
    ) -> None:
        """
        g : node generator object
        k : weight vector of the anisotropy of the multi-index set (TODO: move construction of multi-index outside)
        t : threshold controlling the size of the multi-index set
        d_out : dimension of the output of the target function f
        """
        self.d = len(k)
        self.d_out = d_out
        self._is_nested = g._is_nested

        # Compute coefficients and multi-indices of the Smolyak Operator
        zetas = []
        indxs_all = indices.indexset_sparse(lambda j: k[j], t, cutoff=self.d)
        indxs_zeta = []
        for nu in indxs_all:
            zeta = indices.smolyak_coefficient_zeta_sparse(
                lambda j: k[j], t, nu=nu, cutoff=self.d
            )
            if zeta != 0:
                zetas.append(zeta)
                indxs_zeta.append(nu)

        # Compute base shapes for which `evaluate` will be compiled and sort coeffs and idxs accordingly
        self.n_2_tau = {}
        self.n_2_lambda_n = {}
        self.n_2_zetas = {}
        for zeta, nu in zip(zetas, indxs_zeta):
            tau = tuple(sorted(nu.values(), reverse=True))
            n = len(tau)
            self.n_2_tau[n] = tuple(
                max(tau1, tau2) for tau1, tau2 in zip(self.n_2_tau.get(n, tau), tau)
            )
            self.n_2_lambda_n[n] = self.n_2_lambda_n.get(n, []) + [nu]
            self.n_2_zetas[n] = self.n_2_zetas.get(n, []) + [zeta]

        # Allocate and prefill data
        self.data = {}
        self.__compiledfuncs = {}
        self.offset = 0
        for n in self.n_2_tau.keys():
            if n == 0:
                assert len(self.n_2_zetas[n]) == 1
                self.offset = self.n_2_zetas[n][0]
                continue
            nn = len(self.n_2_zetas[n])  # number of indices in lambda_n
            tau = self.n_2_tau[n]  # multi

            if n not in self.data:
                self.data[n] = {}
            self.data[n]["z"] = jnp.array(self.n_2_zetas[n])
            self.data[n]["F"] = np.zeros(
                (nn, d_out) + tuple(tau_i + 1 for tau_i in tau)
            )
            self.data[n]["xi"] = [np.zeros((nn, tau_i + 1)) for tau_i in tau]
            self.data[n]["w"] = [np.zeros((nn, tau_i + 1)) for tau_i in tau]
            self.data[n]["s"] = np.zeros((nn, n), dtype=int)

            for i, nu in enumerate(self.n_2_lambda_n[n]):
                adims = list(nu.keys())  # active dimensions
                ordering = sorted(
                    range(n), key=lambda i: list(nu.values())[i], reverse=True
                )

                self.data[n]["s"][i] = [adims[o] for o in ordering]

                for t, o in enumerate(ordering):
                    dim = adims[o]
                    nodes = g[dim](nu[dim])
                    self.data[n]["xi"][t][i][: len(nodes)] = nodes
                    self.data[n]["w"][t][i][: len(nodes)] = [
                        1 / np.prod([nj - nk for nk in nodes if nk != nj])
                        for nj in nodes
                    ]

        # Other variables, info, etc
        self.zero = g.get_zero()
        if self.is_nested:
            self.n = len(indxs_all)
        else:
            self.n = int(
                np.sum([np.prod([si + 1 for si in idx.values()]) for idx in indxs_zeta])
            )
        self.n_f_evals = 0

        if f is not None:
            self.set_F(f=f, batchsize=batchsize)

    def set_F(self, *, f: Callable, F: dict = None, batchsize: int = 250) -> dict:
        if F is None:
            F = {}

        # Special case n = 0
        nu = indices.sparse_index_to_dense({}, self.d)
        if self.is_nested:
            Fo = F
        else:
            Fo = F.get(nu, {})  # in this case, idx == degrees
        if nu not in Fo.keys():
            Fo[nu] = f(copy.deepcopy(self.zero))
            self.n_f_evals += 1
        self.offset *= Fo[nu]

        if not self.is_nested:
            F[nu] = Fo

        # n > 0
        for n in self.data.keys():
            for i, nu in enumerate(self.n_2_lambda_n[n]):
                degrees = indices.sparse_index_to_dense(nu, self.d)
                x = copy.deepcopy(self.zero)

                if self.is_nested:
                    Fo = F
                else:
                    Fo = F.get(degrees, {})

                for mu in it.product(*(range(d + 1) for d in degrees)):
                    ridx = tuple(mu[j] for j in self.data[n]["s"][i])
                    if mu not in Fo.keys():
                        for k, (dim, deg) in enumerate(zip(self.data[n]["s"][i], ridx)):
                            x[dim] = self.data[n]["xi"][k][i][deg]
                        Fo[mu] = f(x)
                        self.n_f_evals += 1
                    self.data[n]["F"][i][:, *ridx] = Fo[mu]

                if not self.is_nested:
                    F[degrees] = Fo

            # cast to jnp data structures
            self.data[n]["F"] = jnp.array(self.data[n]["F"])
            self.data[n]["xi"] = [jnp.array(data) for data in self.data[n]["xi"]]
            self.data[n]["w"] = [jnp.array(data) for data in self.data[n]["w"]]
            self.data[n]["s"] = jnp.array(self.data[n]["s"])

        self.__compile_for_batchsize(batchsize)

        return F

    def __compile_for_batchsize(self, batchsize: int) -> None:
        for k in self.data.keys():
            self.__compiledfuncs[k] = jax.vmap(
                _create_evaluate_tensorproduct_interpolant_for_vmap(k),
                in_axes=(None, 0) + (0,) * (2 * k) + (0,),
            )
        _ = self(np.random.random((batchsize, self.d)))

    def __call__(self, x: ArrayLike):
        assert bool(self.__compiledfuncs) == bool(
            self.data
        ), "The operator has not yet been compiled for a target function."
        if x.shape == (self.d,):
            x = x[None, :]
        I_Lambda_x = np.broadcast_to(self.offset, (x.shape[0], self.d_out))
        for n in self.data.keys():
            res = self.__compiledfuncs[n](
                x,
                self.data[n]["F"],
                *self.data[n]["xi"],
                *self.data[n]["w"],
                self.data[n]["s"],
            )
            I_Lambda_x += jnp.tensordot(self.data[n]["z"], res, axes=(0, 0))
        if isinstance(I_Lambda_x, np.ndarray):
            return I_Lambda_x
        return I_Lambda_x.block_until_ready()
