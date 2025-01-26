import sys, copy
import numpy as np
import jax
import jax.numpy as jnp
import itertools as it

sys.path.append('../')
from interpolation import indices

jax.config.update("jax_enable_x64", True)

np.seterr(divide='raise')


def evaluate_jax(x, F, n_list, w_list, j_list) :
    norm = jnp.ones(x.shape[0])
    for i, j in enumerate(j_list):
        b = x[:, [j]] - n_list[i]
        try:
            b = w_list[i] / b
        # Even though the case that one x coordinate coincides with an interpolation node is valid, we handle it via an
        # exception since this saves us the expensive and - in the vast majority of cases - unnecessary check.
        except (ZeroDivisionError, FloatingPointError) :
            rows_with_zero = jnp.any(b == 0, axis=1)
            rows_without_zero = jnp.where(~rows_with_zero)[0]
            rows_with_zero = jnp.where(rows_with_zero)[0]
            b = b.at[rows_with_zero].set(jnp.where(b[rows_with_zero] == 0, 1., 0.))
            b = b.at[rows_without_zero].set(w_list[i] / b[rows_without_zero])
        if i == 0:
            F = jnp.einsum(f'ij,kj...->ik...', b, F)
        else:
            F = jnp.einsum(f'ij,ikj...->ik...', b, F)
        norm *= jnp.sum(b, axis=1)
    return F / norm[:, None]


def create_evaluate_for_vmap(shape) :
    def wrapped_function(x, F, *args) :
        m_list = args[:len(shape)]
        w_list = args[len(shape):2 * len(shape)]
        j_list = args[2 * len(shape)]
        return evaluate_jax(x, F, m_list, w_list, j_list)
    return jax.jit(wrapped_function)


class MultivariateSmolyakBarycentricInterpolator :

    def __init__(self, *, g, k, l, rank, f=None, batchsize=250) :
        self.d = len(k)
        self.is_nested = g.is_nested

        # Compute coefficients and multi-indices of the Smolyak Operator
        coeffs = []
        indxs = []
        indxs_all = indices.indexset_sparse(lambda j : k[j], l, cutoff=self.d)
        for idx in indxs_all :
            c = np.sum([(-1)**e for e in indices.abs_e_sparse(lambda j : k[j], l, nu=idx, cutoff=self.d)])
            if c != 0 :
                coeffs.append(c)
                indxs.append(idx)

        # Compute base shapes for which `evaluate` will be compiled and sort coeffs and idxs accordingly
        self.l2_shape = {}
        self.l2_indxs = {}
        self.l2_coeffs = {}
        for c, idx in zip(coeffs, indxs) :
            shape = tuple(sorted(idx.values(), reverse=True))
            length = len(shape)
            self.l2_shape[length] = tuple(max(nu1, nu2) for nu1, nu2 in zip(self.l2_shape.get(length, shape), shape))
            self.l2_indxs[length] = self.l2_indxs.get(length, []) + [idx]
            self.l2_coeffs[length] = self.l2_coeffs.get(length, []) + [c]

        # Allocate and prefill data
        self.data = {}
        self.compiledfuncs = {}
        self.offset = 0
        for l in self.l2_shape.keys() :
            if l == 0 :
                self.offset = self.l2_coeffs[l]
                continue
            n = len(self.l2_coeffs[l]) # number of shapes with length l
            s = self.l2_shape[l] # max shape

            if l not in self.data : self.data[l] = {}
            self.data[l]['c'] = jnp.array(self.l2_coeffs[l])
            self.data[l]['F'] = np.zeros((n, rank) + tuple(si+1 for si in s))

            self.data[l]['n'] = [np.zeros((n, si+1)) for si in s]
            self.data[l]['w'] = [np.zeros((n, si+1)) for si in s]
            self.data[l]['j'] = np.zeros((n, l), dtype=int)

            for i, idx in enumerate(self.l2_indxs[l]) :
                adims = list(idx.keys())  # active dimensions
                ordering = sorted(range(l), key=lambda i: list(idx.values())[i], reverse=True)

                self.data[l]['j'][i] = [adims[o] for o in ordering]

                for k, o in enumerate(ordering) :
                    dim = adims[o]
                    nodes = g[dim](idx[dim])
                    self.data[l]['n'][k][i][:len(nodes)] = nodes
                    self.data[l]['w'][k][i][:len(nodes)] = [1/np.prod([nj - nk for nk in nodes if nk != nj]) for nj in nodes]

        # Other variables, info, etc
        self.zero = np.squeeze([g[i](0) for i in range(self.d)])
        if self.is_nested :
            self.n = len(indxs_all)
        else :
            self.n = int(np.sum([np.prod([si+1 for si in idx.values()]) for idx in indxs]))
        self.n_f_evals = 0

        if f is not None :
            self.set_F(f=f, batchsize=batchsize)

    def set_F(self, *, f, F=None, i=None, batchsize=250) :
        if F is None : F = {}

        # Special case l = 0
        idx = indices.sparse_index_to_dense({}, self.d)
        if self.is_nested :
            Fo = F
        else :
            Fo = F.get(idx, {}) # in this case, idx == degrees
        if idx not in Fo.keys() :
            Fo[idx] = f(copy.deepcopy(self.zero))
            self.n_f_evals += 1
        self.offset *= Fo[idx]

        if not self.is_nested : F[idx] = Fo

        # l > 0
        for l in self.data.keys() :
            for i, nu in enumerate(self.l2_indxs[l]) :
                degrees = indices.sparse_index_to_dense(nu, self.d)
                x       = copy.deepcopy(self.zero)

                if self.is_nested :
                    Fo = F
                else :
                    Fo = F.get(degrees, {})

                for idx in it.product(*(range(d+1) for d in degrees)) :
                    ridx = tuple(idx[j] for j in self.data[l]['j'][i])
                    if idx not in Fo.keys() :
                        for k, (dim, deg) in enumerate(zip(self.data[l]['j'][i], ridx)) :
                            x[dim] = self.data[l]['n'][k][i][deg]
                        Fo[idx] = f(x)
                        self.n_f_evals += 1
                    self.data[l]['F'][i][:, *ridx] = Fo[idx]

                if not self.is_nested : F[degrees] = Fo

            # cast to jnp data structures
            self.data[l]['F'] = jnp.array(self.data[l]['F'])
            self.data[l]['n'] = [jnp.array(n) for n in self.data[l]['n']]
            self.data[l]['w'] = [jnp.array(n) for n in self.data[l]['w']]
            self.data[l]['j'] = jnp.array(self.data[l]['j'])

        self.compile_for_batchsize(batchsize)

        return F

    def compile_for_batchsize(self, batchsize) :
        for l in self.l2_shape.keys() :
            self.compiledfuncs[l] = jax.vmap(create_evaluate_for_vmap(self.l2_shape[l]),
                                             in_axes=(None, 0) + (0,) * (2 * l) + (0,))
        _ = self(np.random.random((batchsize, self.d)))

    def __call__(self, x) :
        r = self.offset
        for l in self.data.keys() :
            res = self.compiledfuncs[l](x, self.data[l]['F'], *self.data[l]['n'], *self.data[l]['w'], self.data[l]['j'])
            r += jnp.tensordot(self.data[l]['c'], res, axes=(0, 0))
        if len(self.data) == 0  :
            r = np.broadcast_to(r, (x.shape[0], r.shape[0]))
        if isinstance(r, np.ndarray) :
            return r
        return r.block_until_ready()
