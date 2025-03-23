import itertools as it
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from . import barycentric, indices, nodes

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
        self,
        *,
        node_gen: nodes.Generator,
        k: ArrayLike,
        t: float,
        d_out: int,
        f: Callable = None,
        batchsize: int = 250,
    ) -> None:
        """
        node_gen : interpolation node generator object
        k : weight vector of the anisotropy of the multi-index set (TODO: move construction of multi-index outside)
        t : threshold controlling the size of the multi-index set
        f : interpolation target function
        d_out : dimension of the output of the target function f
        """
        self.d = len(k)
        self.d_out = d_out
        self._is_nested = node_gen.is_nested

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
                adims = sorted(
                    nu, key=lambda dim: nu[dim], reverse=True
                )  # sorted active dimensions

                self.data[n]["s"][i] = adims

                for t, dim in enumerate(adims):
                    nodes = node_gen[dim](nu[dim])
                    self.data[n]["xi"][t][i][: len(nodes)] = nodes
                    self.data[n]["w"][t][i][: len(nodes)] = barycentric.compute_weights(
                        nodes
                    )

        # Caching the interpolation node for nu = (0,0,...,0) for reuse in self.set_f
        self.zero = node_gen.get_zero()

        # Tracking number of evaluations of the interpolation target f.
        #   - self.n_f_evals tracks the total number of function evaluations used by the interpolator
        #   - self.n_f_evals_new counts only new function calls.
        # If evaluations are reused across interpolator instances, then likely self.n_f_evals_new < self.n_f_evals
        if self.is_nested:
            self.n_f_evals = len(indxs_all)
        else:
            self.n_f_evals = int(
                np.sum([np.prod([si + 1 for si in idx.values()]) for idx in indxs_zeta])
            )
        self.n_f_evals_new = 0

        if f is not None:
            self.set_f(f=f, batchsize=batchsize)

    def set_f(self, *, f: Callable, f_evals: dict = None, batchsize: int = 250) -> dict:
        """
        Compute (or reuse pre-computed) evaluations of the target function f at the interpolation nodes.
        f : interpolation target function
        f_evals : dictionary mapping interpolation nodes to function evaluations
        batchsize : batchsize of interpolator input, used for pre-compiling __call__
        returns : updated dictionary f_evals containing newly computed interpolation node to evaluation mappings
        """
        if f_evals is None:
            f_evals = {}

        # Special case n = 0
        nu_tuple = indices.sparse_index_to_tuple({})
        if self.is_nested:
            f_evals_nu = f_evals
        else:
            f_evals_nu = f_evals.get(nu_tuple, {})
        if nu_tuple not in f_evals_nu.keys():
            f_evals_nu[nu_tuple] = f(self.zero.copy())
            self.n_f_evals_new += 1
        self.offset *= f_evals_nu[nu_tuple]

        if not self.is_nested:
            f_evals[nu_tuple] = f_evals_nu

        # n > 0
        for n in self.data.keys():
            for i, nu in enumerate(self.n_2_lambda_n[n]):
                x = self.zero.copy()

                nu_tuple = indices.sparse_index_to_tuple(nu)
                if self.is_nested:
                    f_evals_nu = f_evals
                else:
                    f_evals_nu = f_evals.get(nu_tuple, {})

                s_i = self.data[n]["s"][i]
                F_i = self.data[n]["F"][i]
                xi = self.data[n]["xi"]

                sorted_s_i = sorted(s_i)
                for mu_degrees in it.product(*(range(nu[j] + 1) for j in s_i)):
                    mu_tuple = tuple(zip(sorted_s_i, mu_degrees))
                    if mu_tuple not in f_evals_nu:
                        x[s_i] = [
                            xi[k][i][deg]
                            for k, (dim, deg) in enumerate(zip(s_i, mu_degrees))
                        ]
                        f_evals_nu[mu_tuple] = f(x)
                        self.n_f_evals_new += 1
                    F_i[:, *mu_degrees] = f_evals_nu[mu_tuple]

                if not self.is_nested:
                    f_evals[nu_tuple] = f_evals_nu

            # cast to jnp data structures
            self.data[n]["F"] = jnp.array(self.data[n]["F"])
            self.data[n]["xi"] = [jnp.array(data) for data in self.data[n]["xi"]]
            self.data[n]["w"] = [jnp.array(data) for data in self.data[n]["w"]]
            self.data[n]["s"] = jnp.array(self.data[n]["s"])

        self.__compile_for_batchsize(batchsize)

        return f_evals

    def __compile_for_batchsize(self, batchsize: int) -> None:
        for k in self.data.keys():
            self.__compiledfuncs[k] = jax.vmap(
                _create_evaluate_tensorproduct_interpolant_for_vmap(k),
                in_axes=(None, 0) + (0,) * (2 * k) + (0,),
            )
        _ = self(np.random.random((batchsize, self.d)))

    def __call__(self, x: ArrayLike) -> ArrayLike:
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
