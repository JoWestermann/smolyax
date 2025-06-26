import itertools as it
from collections import defaultdict
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from . import barycentric, indices, nodes, quadrature

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
        batchsize: int = None,
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
        batchsize : int, default=None
            Expected batch size of the interpolator input, used to pre-compile the `__call__` method via a warm-up call.
            If `batchsize` is `None`, the warm-up is skipped.
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

        self.__compiled_tensor_product_evaluation = {}
        self.__compiled_tensor_product_gradient = {}

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
            sorted_degs = self.n_2_sorted_degs[n]
            sorted_dims = self.n_2_sorted_dims[n]
            tau = tuple(int(ti) for ti in sorted_degs.max(axis=0))  # per-dimension maximal degree tau_i

            # allocate the array storing the functions evaluations
            self.n_2_F[n] = np.zeros((nn, self.__d_out) + tuple(ti + 1 for ti in tau), dtype=float)

            # allocate arrays for weights and nodes
            nodes_list = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]
            weights_list = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]

            # populate weights and nodes
            # for each slot t, group i's by (dim,deg) so we only gen once
            for t in range(n):
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
                    nodes_list[t][idxs, :L] = pts
                    weights_list[t][idxs, :L] = wts
                # we can do even better if we vectorize node_gen(degrees) for isotropic rules, like GH or Leja

            self.n_2_nodes[n] = nodes_list
            self.n_2_weights[n] = weights_list

    def set_f(
        self,
        *,
        f: Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]],
        f_evals: dict[tuple, dict[tuple, jax.Array]] = None,
        batchsize: int = None,
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
        batchsize : int, default=None
            Expected batch size of the interpolator input, used to pre-compile the `__call__` method via a warm-up call.
            If `batchsize` is `None`, the warm-up is skipped.

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

        def __create_evaluate_tensor_product_interpolant(n: int):
            def __evaluate_tensor_product_interpolant_wrapped(x, F, *args):
                xi_list = args[:n]
                w_list = args[n : 2 * n]
                s_list = args[2 * n]
                nu = args[2 * n + 1]
                zeta = args[2 * n + 2]
                return barycentric.evaluate_tensor_product_interpolant(x, F, xi_list, w_list, s_list, nu, zeta)

            return jax.vmap(
                jax.jit(__evaluate_tensor_product_interpolant_wrapped), in_axes=(None, 0) + (0,) * (2 * n + 3)
            )

        def __create_evaluate_tensor_product_gradient(n: int):
            def __evaluate_tensor_product_gradient_wrapped(x, F, *args):
                xi_list = args[:n]
                w_list = args[n : 2 * n]
                s_list = args[2 * n]
                nu = args[2 * n + 1]
                zeta = args[2 * n + 2]
                return barycentric.evaluate_tensor_product_gradient(x, F, xi_list, w_list, s_list, nu, zeta)

            return jax.vmap(jax.jit(__evaluate_tensor_product_gradient_wrapped), in_axes=(None, 0) + (0,) * (2 * n + 3))

        for n in self.n_2_F.keys():
            self.__compiled_tensor_product_evaluation[n] = __create_evaluate_tensor_product_interpolant(n)
            self.__compiled_tensor_product_gradient[n] = __create_evaluate_tensor_product_gradient(n)

        if batchsize is not None:
            inputs = jax.random.uniform(jax.random.PRNGKey(0), (batchsize, self.__d_in))
            _ = self(inputs)
            _ = self.gradient(inputs)

    def __call__(self, x: Union[jax.Array, np.ndarray], memory_max_GB: float = 4.0) -> jax.Array:
        """@public
        Evaluate the Smolyak operator at points `x`.

        Parameters
        ----------
        x : Union[jax.Array, np.ndarray]
            Points at which to evaluate the Smolyak interpolant of the target function `f`.
            Shape: `(n_points, d_in)` or `(d_in,)`, where `n_points` is the number of evaluation points
            and `d_in` is the dimension of the input domain.
        memory_max_GB : float, optional
            Maximum memory in gigabytes to use during batched evaluation. Controls the batch size
            to stay within this memory limit. Default is 4.0.

        Returns
        -------
        jax.Array
            The interpolant of the target function `f` evaluated at points `x`. Shape: `(n_points, d_out)`
        """
        assert bool(self.__compiled_tensor_product_evaluation) == bool(
            self.n_2_F
        ), "The operator has not yet been compiled for a target function."
        x = jnp.asarray(x)
        if x.shape == (self.__d_in,):
            x = x[None, :]
        assert x.shape[1] == self.__d_in, f"{x.shape[1]} != {self.__d_in}"

        I_Lambda_x = jnp.broadcast_to(self.offset, (x.shape[0], self.__d_out))

        for n in self.__compiled_tensor_product_evaluation.keys():

            # determine the number of batches that ensures that the computation stays within the given memory limit
            n_terms = self.n_2_F[n].shape[0]
            memory_per_term_GB = I_Lambda_x.size * np.prod(self.n_2_F[n].shape[3:]) * 8 / (1024**3)
            batch_size = max(1, int(np.floor(memory_max_GB / memory_per_term_GB)))
            n_batches = int(np.ceil(n_terms / batch_size))

            # batched processing of tensor product interpolants with n active dimensions
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_terms)
                res = self.__compiled_tensor_product_evaluation[n](
                    x,
                    self.n_2_F[n][start:end],
                    *[arr[start:end] for arr in self.n_2_nodes[n]],
                    *[arr[start:end] for arr in self.n_2_weights[n]],
                    self.n_2_sorted_dims[n][start:end],
                    self.n_2_sorted_degs[n][start:end],
                    self.n_2_zetas[n][start:end],
                )
                I_Lambda_x += jnp.sum(res, axis=0)

        return I_Lambda_x

    def gradient(self, x: Union[jax.Array, np.ndarray], memory_max_GB: float = 4.0) -> jax.Array:
        """
        Compute the gradient of the Smolyak interpolant at the given points.

        Parameters
        ----------
        x : Union[jax.Array, numpy.ndarray]
            Points at which to evaluate the gradient. Shape: `(n_points, d_in)`.
        memory_max_GB : float, optional
            Maximum memory in gigabytes to use during batched evaluation. Controls the batch size
            to stay within this memory limit. Default is 4.0.

        Returns
        -------
        jax.Array
            Gradient of the interpolant evaluated at `x`.
            Shape: `(n_points, d_out, d_in)`.
        """
        assert bool(self.__compiled_tensor_product_evaluation) == bool(
            self.n_2_F
        ), "The operator has not yet been compiled for a target function."
        x = jnp.asarray(x)
        if x.shape == (self.__d_in,):
            x = x[None, :]
        assert x.shape[1] == self.__d_in, f"{x.shape[1]} != {self.__d_in}"

        J_Lambda_x = jnp.zeros((x.shape[0], self.__d_out, self.__d_in))

        for n in self.__compiled_tensor_product_gradient.keys():

            # determine the number of batches that ensures that the computation stays within the given memory limit
            n_terms = self.n_2_F[n].shape[0]
            memory_per_term_GB = J_Lambda_x.size * np.prod(self.n_2_F[n].shape[3:]) * 8 / (1024**3)
            batch_size = max(1, int(np.floor(memory_max_GB / memory_per_term_GB)))
            n_batches = int(np.ceil(n_terms / batch_size))

            # batched processing of tensor product gradients with n active dimensions
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_terms)
                res = self.__compiled_tensor_product_gradient[n](
                    x,
                    self.n_2_F[n][start:end],
                    *[arr[start:end] for arr in self.n_2_nodes[n]],
                    *[arr[start:end] for arr in self.n_2_weights[n]],
                    self.n_2_sorted_dims[n][start:end],
                    self.n_2_sorted_degs[n][start:end],
                    self.n_2_zetas[n][start:end],
                )
                J_Lambda_x += jnp.sum(res, axis=0)

        return J_Lambda_x

    def integral(self) -> jax.Array:
        """
        Compute the integral of the Smolyak interpolant. Note that this is equivalent to a Smolyak quadrature
        approximation to the integral of the target function `f`.

        Returns
        -------
        jax.Array
            Integral of the interpolant. Shape: `(d_out,)`.
        """
        # assemble quadrature weights, closely following the logic in __init_nodes_and_weights
        # ----------------------------------------------------------------------------
        n_2_quad_weights = {}

        for n, nus in self.n_2_nus.items():
            if n == 0:
                continue

            nn = len(nus)  # number of multi-indices of length n
            sorted_degs = self.n_2_sorted_degs[n]
            sorted_dims = self.n_2_sorted_dims[n]
            tau = tuple(int(ti) for ti in sorted_degs.max(axis=0))  # per-dimension maximal degree tau_i

            weights_list = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]
            for t in range(n):
                groups: dict[tuple[int, int], list[int]] = defaultdict(list)
                for i in range(nn):
                    dim = int(sorted_dims[i, t])
                    deg = int(sorted_degs[i, t])
                    groups[(dim, deg)].append(i)
                for (dim, deg), idxs in groups.items():
                    wts = self.__node_gen[dim].get_quadrature_weights(deg)
                    L = len(wts)
                    weights_list[t][idxs, :L] = wts
            n_2_quad_weights[n] = [jnp.array(w) for w in weights_list]

        # jit compile and evaluate tensor product terms
        # ----------------------------------------------------------------------------
        def __create_evaluate_tensor_product_quadrature(n: int):
            def __evaluate_tensor_product_quadrature_wrapped(F, *w_list):
                return quadrature.evaluate_tensor_product_quadrature(F, w_list)

            return jax.vmap(jax.jit(__evaluate_tensor_product_quadrature_wrapped), in_axes=(0,) * (n + 1))

        Q_Lambda = jnp.broadcast_to(self.offset, self.__d_out)
        for n in self.n_2_F.keys():
            quadrature_func_n = __create_evaluate_tensor_product_quadrature(n)

            res = quadrature_func_n(self.n_2_F[n], *n_2_quad_weights[n])

            Q_Lambda += jnp.tensordot(self.n_2_zetas[n], res, axes=1)
        return Q_Lambda.block_until_ready()
