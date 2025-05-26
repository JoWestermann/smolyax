import itertools as it
from collections import defaultdict
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from . import barycentric, indices, nodes

jax.config.update("jax_enable_x64", True)


class SmolyakBarycentricInterpolator:
    """
    A class implementing the Smolyak operator to interpolate high-dimensional and vector-valued functions.
    """

    @property
    def d_in(self) -> int:
        """Input dimension of target function and interpolant"""
        return self.__d_in

    @property
    def d_out(self) -> int:
        """Output dimension of target function and interpolant"""
        return self.__d_out

    def __init__(
        self,
        *,
        node_gen: nodes.Generator,
        k: Sequence[float],
        t: float,
        d_out: int,
        f: Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]] = None,
        batchsize: int = 250,
    ) -> None:
        """
        Initialize the Smolyak Barycentric Interpolator.

        Parameters
        ----------
        node_gen : nodes.Generator
            Generator object that returns interpolation nodes for each dimension.
        k : Union[jax.Array, np.ndarray]
            Anisotropy weight vector of the multi-index set. Shape `(d_in,)`.
        t : float
            Threshold that controls the size of the multi-index set.
        d_out : int
            Output dimension of the target function.
        f : Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]], optional
            Target function to interpolate. While `f` can be passed at construction time, for a better control over, and
            potential reuse of, function evaluations consider calling [`set_f()`](#SmolyakBarycentricInterpolator.set_f)
            *after* construction.
        batchsize : int, default=250
            Anticipated batch size of the interpolator input, used for pre-compiling the `__call__` method.
        """
        self.__d_in = len(k)
        self.__d_out = d_out
        self.__is_nested = node_gen.is_nested
        self.__node_gen = node_gen

        # Step 1 : Bin multiindices and smolyak coefficients by the number of active dimensions, n
        self.__init_indices_data(k, t)

        # Step 2 : Compute sorted dimensions and sorted degrees for all multi-indices
        self.__init_indices_sorting()

        # Step 3 : Allocate and prefill data
        self.__init_nodes_and_weights()

        # Caching the interpolation node for nu = (0,0,...,0) for reuse in self.set_f
        self.__zero = np.array([g(0)[0] for g in self.__node_gen])

        self.__compiledfuncs = {}

        if f is not None:
            self.set_f(f=f, batchsize=batchsize)

    def __init_indices_data(self, k: Sequence[float], t: float):
        self.n_2_nus, self.n_2_zetas = indices.non_zero_indices_and_zetas(k, t)

        # Tracking number of evaluations of the interpolation target f.
        #   - self.n_f_evals tracks the total number of function evaluations used by the interpolator
        #   - self.n_f_evals_new counts only new function calls.
        # If evaluations are reused across interpolator instances, then likely self.n_f_evals_new < self.n_f_evals

        self.n_f_evals = indices.nodeset_cardinality(k, t, nested=self.__is_nested)
        self.n_f_evals_new = 0

    def __init_indices_sorting(self):
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
                self.n_2_dims[n][i] = list(k for k, _ in nu)
                sorted_nu = sorted(nu, key=lambda x: x[1], reverse=True)
                self.n_2_sorted_dims[n][i], self.n_2_sorted_degs[n][i] = zip(*sorted_nu)
                self.n_2_argsort_dims[n][i] = np.argsort(self.n_2_sorted_dims[n][i])

    def __build_nodes_weights(self, sorted_dims, sorted_degs):
        """
        Build lists of per-slot (nn, tau_i+1) arrays of nodes & weights.

        Parameters
        ----------
        sorted_dims : np.ndarray[int64] shape (nn, n_dims)
        sorted_degs : np.ndarray[int64] shape (nn, n_dims)

        Returns
        -------
        nodes_list   : list of n_dims arrays, each shape (nn, tau_i+1)
        weights_list : same shape, filled with barycentric weights
        """
        nn, n_dims = sorted_dims.shape

        # compute per-slot max degree tau_i
        tau = sorted_degs.max(axis=0).astype(int)  # we might want to just keep track of this during construction!

        # pre-allocate output arrays
        nodes_list = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]
        weights_list = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]

        # for each slot t, group i's by (dim,deg) so we only gen once
        for t in range(n_dims):
            groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for i in range(nn):
                dim = int(sorted_dims[i, t])
                deg = int(sorted_degs[i, t])
                groups[(dim, deg)].append(i)

            # now for each unique (dim,deg) compute pts & wts once
            for (dim, deg), idxs in groups.items():
                pts = self.__node_gen[dim](deg)
                wts = barycentric.compute_weights(pts)
                L = len(pts)
                # Better memory access
                nodes_list[t][idxs, :L] = pts
                weights_list[t][idxs, :L] = wts
            # we can do even better if we vectorize node_gen(degrees) for isotropic rules, like GH or Leja
        return nodes_list, weights_list

    def __init_nodes_and_weights(self):
        self.offset = 0
        self.n_2_F = {}
        self.n_2_nodes = {}
        self.n_2_weights = {}

        for n, nus in self.n_2_nus.items():
            if n == 0:
                # Smolyak constant term
                assert len(self.n_2_zetas[n]) == 1
                self.offset = self.n_2_zetas[n][0]
                continue

            nn = len(nus)  # number of multi-indices of length n
            # per-slot max degree tau_i
            sorted_degs = np.array(self.n_2_sorted_degs[n], dtype=int)
            tau = tuple(int(ti) for ti in sorted_degs.max(axis=0))

            # allocate the F array
            self.n_2_F[n] = np.zeros((nn, self.__d_out) + tuple(ti + 1 for ti in tau), dtype=float)

            # build  nodes & weights via slicing
            self.n_2_nodes[n], self.n_2_weights[n] = self.__build_nodes_weights(
                np.array(self.n_2_sorted_dims[n], dtype=int), sorted_degs
            )

    def set_f(
        self,
        *,
        f: Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]],
        f_evals: dict[tuple, dict[tuple, jax.Array]] = None,
        batchsize: int = 250,
    ) -> dict[tuple, dict[tuple, jax.Array]]:
        """
        Compute (or reuse pre-computed) evaluations of the target function `f` at the interpolation nodes of the
        Smolyak operator.

        Parameters
        ----------
        f : Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]]
            Target function to interpolate.
        f_evals : dict, optional
            A dictionary mapping interpolation nodes to function evaluations.
            If provided, these evaluations will be reused.
        batchsize : int, default=250
            Anticipated batch size of the interpolator input, used for pre-compiling the `__call__` method.

        Returns
        -------
        dict
            An updated dictionary containing all computed evaluations of the target function `f`.
        """
        if f_evals is None:
            f_evals = {}

        # Special case n = 0
        nu = ()
        if self.__is_nested:
            f_evals_nu = f_evals
        else:
            f_evals_nu = f_evals.get(nu, {})
        if nu not in f_evals_nu.keys():
            f_evals_nu[nu] = f(self.__zero.copy())
            self.n_f_evals_new += 1
        self.offset *= f_evals_nu[nu]

        if not self.__is_nested:
            f_evals[nu] = f_evals_nu

        # n > 0
        for n in self.n_2_F.keys():
            nodes = self.n_2_nodes[n]
            for i, nu in enumerate(self.n_2_nus[n]):
                x = self.__zero.copy()

                if self.__is_nested:
                    f_evals_nu = f_evals
                else:
                    f_evals_nu = f_evals.get(nu, {})

                s_i = self.n_2_sorted_dims[n][i]
                argsort_s_i = self.n_2_argsort_dims[n][i]
                F_i = self.n_2_F[n][i]

                ranges = [range(k + 1) for k in self.n_2_sorted_degs[n][i]]
                for mu_degrees in it.product(*ranges):
                    mu_tuple = tuple((s_i[i], mu_degrees[i]) for i in argsort_s_i if mu_degrees[i] > 0)
                    if mu_tuple not in f_evals_nu:
                        x[s_i] = [xi_k[i][deg] for xi_k, deg in zip(nodes, mu_degrees)]
                        f_evals_nu[mu_tuple] = f(x)
                        self.n_f_evals_new += 1
                    F_i[:, *mu_degrees] = f_evals_nu[mu_tuple]

                if not self.__is_nested:
                    f_evals[nu] = f_evals_nu

            # cast to jnp data structures
            self.n_2_F[n] = jnp.array(self.n_2_F[n])
            self.n_2_nodes[n] = [jnp.array(xi) for xi in self.n_2_nodes[n]]
            self.n_2_weights[n] = [jnp.array(w) for w in self.n_2_weights[n]]
            self.n_2_sorted_dims[n] = jnp.array(self.n_2_sorted_dims[n])
            self.n_2_zetas[n] = jnp.array(self.n_2_zetas[n])

        self.__compile_for_batchsize(batchsize)

        return f_evals

    def __compile_for_batchsize(self, batchsize: int) -> None:

        compute_basis = barycentric.evaluate_basis_numerator_centered
        if not (self.__zero == 0.0).all():
            compute_basis = barycentric.evaluate_basis_numerator_noncentered

        def __evaluate_tensorproduct_interpolant(
            x: jax.Array,
            F: jax.Array,
            xi_list: Sequence[jax.Array],
            w_list: Sequence[jax.Array],
            sorted_dims: Sequence[int],
            sorted_degs: Sequence[int],
        ) -> jax.Array:
            """
            Evaluate a tensor product interpolant.

            Parameters
            ----------
            x : jax.Array
                Points at which to evaluate the tensor product interpolant of the target function `f`.
                Should be a 2D array of shape `(n_points, d_in)` where `n_points` is the number of evaluation points
                and `d_in` is the dimension of the input domain.

            F : jax.Array
                Tensors storing the evaluations of the target function `f`.
                Should be a multi-dimensional array with shape `(d_out, mu_1, mu_2, ..., mu_n)`
                where each `mu_i` corresponds to the number of points in the ith dimension.

            xi_list : Sequence[jax.Array]
                Interpolation nodes. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

            w_list : Sequence[jax.Array]
                Interpolation weights. A sequence of 1D arrays, each with shape `(mu_i,)` for the ith dimension.

            sorted_dims : Sequence[int]
                Dimensions with nonzero interpolation degree.

            sorted_degs : Sequence[int]
                Interpolation degrees per dimension.

            Returns
            -------
            jax.Array
                The evaluated tensor product interpolant at the points specified by `x`.
                The shape of the output will be `(n_points, d_out)`.
            """
            norm = jnp.ones(x.shape[0])
            for i, (si, nui) in enumerate(zip(sorted_dims, sorted_degs)):
                b = compute_basis(x[:, [si]], xi_list[i], w_list[i], nui)
                if i == 0:
                    F = jnp.einsum("ij,kj...->ik...", b, F)
                else:
                    F = jnp.einsum("ij,ikj...->ik...", b, F)
                norm *= jnp.sum(b, axis=1)
            return F / norm[:, None]

        def __create_evaluate_tensorproduct_interpolant_for_vmap(n: int):
            """
            Create a JIT-compiled function for evaluating a tensor product interpolant, for use with `jax.vmap`.

            Parameters
            ----------
            n : int
                The number of dimensions in the tensor product interpolant. This determines how the input arguments are
                split into `xi_list`, `w_list`, and `s_list`.

            Returns
            -------
            Callable
                A JIT-compiled function of __evaluate_tensorproduct_interpolant.
            """

            def __evaluate_tensorproduct_interpolant_wrapped(x, F, *args):
                xi_list = args[:n]
                w_list = args[n : 2 * n]
                s_list = args[2 * n]
                nu = args[2 * n + 1]
                return __evaluate_tensorproduct_interpolant(x, F, xi_list, w_list, s_list, nu)

            return jax.jit(__evaluate_tensorproduct_interpolant_wrapped)

        for n in self.n_2_F.keys():
            self.__compiledfuncs[n] = jax.vmap(
                __create_evaluate_tensorproduct_interpolant_for_vmap(n),
                in_axes=(None, 0) + (0,) * (2 * n) + (0, 0),
            )
        _ = self(jax.random.uniform(jax.random.PRNGKey(0), (batchsize, self.__d_in)))

    def __call__(self, x: Union[jax.Array, np.ndarray]) -> jax.Array:
        """@public
        Evaluate the Smolyak operator at points `x`.

        Parameters
        ----------
        x : Union[jax.Array, np.ndarray]
            Points at which to evaluate the Smolyak interpolant of the target function `f`.
            Shape: `(n_points, d_in)` or `(d_in,)`, where `n_points` is the number of evaluation points
            and `d_in` is the dimension of the input domain.

        Returns
        -------
        jax.Array
            The interpolant of the target function `f` evaluated at points `x`. Shape: `(n_points, d_out)`
        """
        assert bool(self.__compiledfuncs) == bool(
            self.n_2_F
        ), "The operator has not yet been compiled for a target function."
        x = jnp.asarray(x)
        if x.shape == (self.__d_in,):
            x = x[None, :]
        I_Lambda_x = jnp.broadcast_to(self.offset, (x.shape[0], self.__d_out))
        for n in self.__compiledfuncs.keys():
            res = self.__compiledfuncs[n](
                x,
                self.n_2_F[n],
                *self.n_2_nodes[n],
                *self.n_2_weights[n],
                self.n_2_sorted_dims[n],
                self.n_2_sorted_degs[n],
            )
            I_Lambda_x += jnp.tensordot(self.n_2_zetas[n], res, axes=(0, 0))
        return I_Lambda_x.block_until_ready()
