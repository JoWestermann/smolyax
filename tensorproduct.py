import numpy as np
import itertools as it

import points, indices

np.seterr(divide='raise')


class TensorProductBarycentricInterpolator:
    """
    Class that implements the multivariate barycentric interpolation formula of the second form, see
    [Klimke, Uncertainty Modeling using Fuzzy Arithmetic and Sparse Grids (Section 3, in particular equ. 3.35)]
    (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9da00c0aedf5335927147df78eb5f6767edba968).
    """

    def __init__(self, gens, degrees, d, f=None) :
        self.gens    = gens
        self.adims   = list(degrees.keys())
        self.nodes   = [gens[i](k) for i, k in degrees.items()]
        self.weights = [np.array([1/np.prod([nj - nk for nk in nodes if nk != nj]) for nj in nodes]) for nodes in self.nodes]
        self.degrees = indices.sparse_index_to_dense(degrees, d)
        self.x       = np.squeeze(np.array([gens[i](0) for i in range(gens.d)]))
        # F is the array containing evaluations of the target f
        # NOTE: For efficiency, we index only dimensions that have a degree larger than 0.
        self.ordering = np.argsort([len(nodes) for nodes in self.nodes])[::-1]
        self.F = np.zeros(tuple(len(self.nodes[o]) for o in self.ordering))
        if f is not None :
            self.set_F(f)

    def set_x(self, ridx) :
        for o,i in zip(self.ordering, ridx) :
            self.x[self.adims[o]] = self.nodes[o][i]

    def set_F(self, f) :
        """Evaluate the target function f at all interpolation nodes"""
        for ridx in it.product(*(range(d) for d in self.F.shape)) :
            self.set_x(ridx)
            self.F[ridx] = f(self.x)

    def reduced_index(self, idx) :
        return tuple(idx[self.adims[o]] for o in self.ordering)

    def __call__(self, x) :
        assert x.ndim <= 2
        assert x.shape[x.ndim-1] == len(self.degrees)

        # Special case for interpolation degrees (0,0,...,0)
        if self.F.ndim == 0 :
            if x.ndim == 1 :
                return self.F
            return np.ones((x.shape[0],)) * self.F

        # Initializing the result for the nontrivial multivariate case
        res = self.F

        # Evaluating the interpolator at a single multivariate point x
        if x.ndim == 1 :
            assert len(x) == len(self.degrees)
            for o in self.ordering :
                b = x[self.adims[o]] - self.nodes[o]
                try :
                    b = self.weights[o] / b
                except (ZeroDivisionError, FloatingPointError) :
                    b = np.where(b == 0, 1., 0.)
                res = np.einsum(f'i,i...->...', b, res)
                res /= np.sum(b)
            return res

        # Efficient implementation for evaluating the interpolator for n multivariate points simultaneously
        assert x.shape[1] == len(self.degrees)
        norm = np.ones(x.shape[0])
        for i, o in enumerate(self.ordering) :
            b = x[:, [self.adims[o]]] - self.nodes[o]
            try :
                b = self.weights[o] / b
            except (ZeroDivisionError, FloatingPointError) :
                rows_with_zero       = np.any(b == 0, axis=1)
                rows_without_zero    = np.where(~rows_with_zero)[0]
                rows_with_zero       = np.where(rows_with_zero)[0]
                b[rows_with_zero]    = np.where(b[rows_with_zero] == 0, 1., 0.)
                b[rows_without_zero] = self.weights[o] / b[rows_without_zero]
            if i == 0 :
                res = np.einsum(f'ij,j...->i...', b, res)
            else :
                res = np.einsum(f'ij,ij...->i...', b, res)
            norm *= np.sum(b, axis=1)
        return res / norm


