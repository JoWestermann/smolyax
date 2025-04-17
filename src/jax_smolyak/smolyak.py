import itertools as it
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from . import barycentric, indices, nodes

jax.config.update("jax_enable_x64", True)


class SmolyakBarycentricInterpolator:

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

        indxs_all = indices.indexset(k, t)

        # Step 1 : Sort multiindices and smolyak coefficients by the number of active dimensions, n

        indxs_zeta = []
        self.n_2_nus = {}
        self.n_2_zetas = {}

        for nu in indxs_all:
            zeta = indices.smolyak_coefficient_zeta(k, t, nu=nu)
            if zeta != 0:
                indxs_zeta.append(nu)
                n = len(nu)
                self.n_2_nus[n] = self.n_2_nus.get(n, []) + [nu]
                self.n_2_zetas[n] = self.n_2_zetas.get(n, []) + [zeta]

        # Step 2 : Compute sorted dimensions and sorted degrees for all multi-indices

        self.n_2_dims = {}
        self.n_2_sorted_dims = {}
        self.n_2_sorted_degs = {}
        self.n_2_argsort_dims = {}

        for n in self.n_2_nus.keys():
            if n == 0:
                continue
            nn = len(self.n_2_nus[n])
            self.n_2_dims[n] = np.empty((nn, n), dtype=int)
            self.n_2_sorted_dims[n] = np.empty((nn, n), dtype=int)
            self.n_2_sorted_degs[n] = np.empty((nn, n), dtype=int)
            self.n_2_argsort_dims[n] = np.empty((nn, n), dtype=int)

            for i, nu in enumerate(self.n_2_nus[n]):
                self.n_2_dims[n][i] = list(nu.keys())
                sorted_nu = sorted(nu.items(), key=lambda x: x[1], reverse=True)
                self.n_2_sorted_dims[n][i], self.n_2_sorted_degs[n][i] = zip(*sorted_nu)
                self.n_2_argsort_dims[n][i] = np.argsort(self.n_2_sorted_dims[n][i])

        # Step 3 : Allocate and prefill data

        self.offset = 0
        self.n_2_F = {}
        self.n_2_nodes = {}
        self.n_2_weights = {}
        for n in self.n_2_nus.keys():

            if n == 0:
                assert len(self.n_2_zetas[n]) == 1
                self.offset = self.n_2_zetas[n][0]
                continue

            nn = len(self.n_2_nus[n])  # number of indices in lambda_n
            tau = np.max(self.n_2_sorted_degs[n], axis=0)

            self.n_2_F[n] = np.zeros((nn, d_out) + tuple(tau_i + 1 for tau_i in tau))
            self.n_2_nodes[n] = [np.zeros((nn, tau_i + 1)) for tau_i in tau]
            self.n_2_weights[n] = [np.zeros((nn, tau_i + 1)) for tau_i in tau]

            for i, nu in enumerate(self.n_2_nus[n]):
                for t, dim in enumerate(self.n_2_sorted_dims[n][i]):
                    nodes = node_gen[dim](nu[dim])
                    self.n_2_nodes[n][t][i][: len(nodes)] = nodes
                    self.n_2_weights[n][t][i][: len(nodes)] = barycentric.compute_weights(nodes)

        # Caching the interpolation node for nu = (0,0,...,0) for reuse in self.set_f
        self.zero = node_gen.get_zero()

        # Tracking number of evaluations of the interpolation target f.
        #   - self.n_f_evals tracks the total number of function evaluations used by the interpolator
        #   - self.n_f_evals_new counts only new function calls.
        # If evaluations are reused across interpolator instances, then likely self.n_f_evals_new < self.n_f_evals
        if self.is_nested:
            self.n_f_evals = len(indxs_all)
        else:
            self.n_f_evals = int(np.sum([np.prod([si + 1 for si in idx.values()]) for idx in indxs_zeta]))
        self.n_f_evals_new = 0

        self.__compiledfuncs = {}

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
        for n in self.n_2_F.keys():
            nodes = self.n_2_nodes[n]
            for i, nu in enumerate(self.n_2_nus[n]):
                x = self.zero.copy()

                nu_tuple = indices.sparse_index_to_tuple(nu)
                if self.is_nested:
                    f_evals_nu = f_evals
                else:
                    f_evals_nu = f_evals.get(nu_tuple, {})

                s_i = self.n_2_sorted_dims[n][i]
                argsort_s_i = self.n_2_argsort_dims[n][i]
                F_i = self.n_2_F[n][i]

                ranges = [range(nu[j] + 1) for j in s_i]
                for mu_degrees in it.product(*ranges):
                    mu_tuple = tuple((s_i[i], mu_degrees[i]) for i in argsort_s_i if mu_degrees[i] > 0)
                    if mu_tuple not in f_evals_nu:
                        x[s_i] = [xi_k[i][deg] for xi_k, deg in zip(nodes, mu_degrees)]
                        f_evals_nu[mu_tuple] = f(x)
                        self.n_f_evals_new += 1
                    F_i[:, *mu_degrees] = f_evals_nu[mu_tuple]

                if not self.is_nested:
                    f_evals[nu_tuple] = f_evals_nu

            # cast to jnp data structures
            self.n_2_F[n] = jnp.array(self.n_2_F[n])
            self.n_2_nodes[n] = [jnp.array(xi) for xi in self.n_2_nodes[n]]
            self.n_2_weights[n] = [jnp.array(w) for w in self.n_2_weights[n]]
            self.n_2_sorted_dims[n] = jnp.array(self.n_2_sorted_dims[n])
            self.n_2_zetas[n] = jnp.array(self.n_2_zetas[n])

        self.__compile_for_batchsize(batchsize)

        return f_evals

    def __compile_for_batchsize(self, batchsize: int) -> None:

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

        for n in self.n_2_F.keys():
            self.__compiledfuncs[n] = jax.vmap(
                _create_evaluate_tensorproduct_interpolant_for_vmap(n),
                in_axes=(None, 0) + (0,) * (2 * n) + (0,),
            )
        _ = self(np.random.random((batchsize, self.d)))

    def __call__(self, x: ArrayLike) -> ArrayLike:
        assert bool(self.__compiledfuncs) == bool(
            self.n_2_F
        ), "The operator has not yet been compiled for a target function."
        if x.shape == (self.d,):
            x = x[None, :]
        I_Lambda_x = np.broadcast_to(self.offset, (x.shape[0], self.d_out))
        for n in self.__compiledfuncs.keys():
            res = self.__compiledfuncs[n](
                x,
                self.n_2_F[n],
                *self.n_2_nodes[n],
                *self.n_2_weights[n],
                self.n_2_sorted_dims[n],
            )
            I_Lambda_x += jnp.tensordot(self.n_2_zetas[n], res, axes=(0, 0))
        if isinstance(I_Lambda_x, np.ndarray):
            return I_Lambda_x
        return I_Lambda_x.block_until_ready()
