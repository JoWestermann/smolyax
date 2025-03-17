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
    x: ArrayLike, F: ArrayLike, n_list: List, w_list: List, t_list: List
) -> ArrayLike:
    norm = jnp.ones(x.shape[0])
    for i, ti in enumerate(t_list):
        b = x[:, [ti]] - n_list[i]
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


def _create_evaluate_tensorproduct_interpolant_for_vmap(k: int):
    def wrapped_function(x, F, *args):
        m_list = args[:k]
        w_list = args[k:2 * k]
        j_list = args[2 * k]
        return __evaluate_tensorproduct_interpolant(x, F, m_list, w_list, j_list)

    return jax.jit(wrapped_function)


class MultivariateSmolyakBarycentricInterpolator:

    @property
    def is_nested(self) -> bool:
        return self._is_nested

    def __init__(
        self, *, g, k, l, rank: int, f: Callable = None, batchsize: int = 250
    ) -> None:
        self.d = len(k)
        self.d_out = rank
        self._is_nested = g._is_nested

        # Compute coefficients and multi-indices of the Smolyak Operator
        zetas = []
        indxs_all = indices.indexset_sparse(lambda j: k[j], l, cutoff=self.d)
        indxs_zeta = []
        for idx in indxs_all:
            zeta = indices.smolyak_coefficient_zeta_sparse(
                lambda j: k[j], l, nu=idx, cutoff=self.d
            )
            if zeta != 0:
                zetas.append(zeta)
                indxs_zeta.append(idx)

        # Compute base shapes for which `evaluate` will be compiled and sort coeffs and idxs accordingly
        self.k_2_tau = {}
        self.k_2_lambda_k = {}
        self.k_2_zetas = {}
        for zeta, nu in zip(zetas, indxs_zeta):
            tau = tuple(sorted(nu.values(), reverse=True))
            kk = len(tau)
            self.k_2_tau[kk] = tuple(
                max(nu1, nu2) for nu1, nu2 in zip(self.k_2_tau.get(kk, tau), tau)
            )
            self.k_2_lambda_k[kk] = self.k_2_lambda_k.get(kk, []) + [nu]
            self.k_2_zetas[kk] = self.k_2_zetas.get(kk, []) + [zeta]

        # Allocate and prefill data
        self.data = {}
        self.__compiledfuncs = {}
        self.offset = 0
        for k in self.k_2_tau.keys():
            if k == 0:
                assert len(self.k_2_zetas[k]) == 1
                self.offset = self.k_2_zetas[k][0]
                continue
            n = len(self.k_2_zetas[k])  # number of shapes with length k
            s = self.k_2_tau[k]  # max shape

            if k not in self.data:
                self.data[k] = {}
            self.data[k]["z"] = jnp.array(self.k_2_zetas[k])
            self.data[k]["F"] = np.zeros((n, rank) + tuple(si + 1 for si in s))
            self.data[k]["n"] = [np.zeros((n, si + 1)) for si in s]
            self.data[k]["w"] = [np.zeros((n, si + 1)) for si in s]
            self.data[k]["t"] = np.zeros((n, k), dtype=int)

            for i, idx in enumerate(self.k_2_lambda_k[k]):
                adims = list(idx.keys())  # active dimensions
                ordering = sorted(
                    range(k), key=lambda i: list(idx.values())[i], reverse=True
                )

                self.data[k]["t"][i] = [adims[o] for o in ordering]

                for t, o in enumerate(ordering):
                    dim = adims[o]
                    nodes = g[dim](idx[dim])
                    self.data[k]["n"][t][i][: len(nodes)] = nodes
                    self.data[k]["w"][t][i][: len(nodes)] = [
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

        # Special case l = 0
        idx = indices.sparse_index_to_dense({}, self.d)
        if self.is_nested:
            Fo = F
        else:
            Fo = F.get(idx, {})  # in this case, idx == degrees
        if idx not in Fo.keys():
            Fo[idx] = f(copy.deepcopy(self.zero))
            self.n_f_evals += 1
        self.offset *= Fo[idx]

        if not self.is_nested:
            F[idx] = Fo

        # l > 0
        for l in self.data.keys():
            data_l = self.data[l]
            data_l_t = self.data[l]["t"]
            data_l_n = self.data[l]["n"]
            for i, nu in enumerate(self.k_2_lambda_k[l]):
                degrees = indices.sparse_index_to_dense(nu, self.d)
                x = copy.deepcopy(self.zero)

                data_l_t_i = data_l_t[i]
                #should store ridx as well!, data[l][t_inv]
                data_l_f_i = data_l["F"][i]

                if self.is_nested:
                    Fo = F
                else:
                    Fo = F.get(degrees, {})
                print(len(Fo.keys()),"fo.keys length")
                for idx in it.product(*(range(d + 1) for d in degrees)): #can we loop not through all indices, but only 
                    ridx = tuple(idx[j] for j in data_l_t_i) #ridx can be known a prior as a function of data_l_t_i and a sparse idx 
                    # print(idx, data_l_t_i, (tuple(ridx))) #t = 1235 maps to the same ridx as t=1234, the key here is that one can construct it without explicit access to 0's (0's can be implicit
                    
                    if idx not in Fo.keys():
                        for k, (dim, deg) in enumerate(zip(data_l_t_i, ridx)):
                            x[dim] = data_l_n[k][i][deg]
                        Fo[idx] = f(x)
                        self.n_f_evals += 1
                    data_l_f_i[:, *ridx] = Fo[idx] #idx can be sparse,rather than dense! 
                if not self.is_nested:
                    F[degrees] = Fo
            # cast to jnp data structures
            self.data[l]["F"] = jnp.asarray(self.data[l]["F"])
            self.data[l]["n"] = [jnp.asarray(n) for n in self.data[l]["n"]]
            self.data[l]["w"] = [jnp.asarray(n) for n in self.data[l]["w"]]
            self.data[l]["t"] = jnp.asarray(self.data[l]["t"])

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
        for k in self.data.keys():
            res = self.__compiledfuncs[k](
                x,
                self.data[k]["F"],
                *self.data[k]["n"],
                *self.data[k]["w"],
                self.data[k]["t"],
            )
            I_Lambda_x += jnp.tensordot(self.data[k]["z"], res, axes=(0, 0))
        if isinstance(I_Lambda_x, np.ndarray):
            return I_Lambda_x
        return I_Lambda_x.block_until_ready()
