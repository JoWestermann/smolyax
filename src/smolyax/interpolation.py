import itertools as it
import string
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

    @property
    def n_f_evals(self) -> int:
        """Number of function evaluations (== number of interpolation nodes) used by the interpolator"""
        return self.__n_f_evals

    @property
    def n_f_evals_new(self) -> int:
        """Number of function evaluations that were not reused from previous computations"""
        return self.__n_f_evals_new

    def __init__(
        self,
        *,
        d_out: int,
        node_gen: nodes.Generator,
        k: Sequence[float],
        t: float,
        f: Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]] = None,
        n_inputs: int = None,
        memory_limit: float = 4.0,
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
        n_inputs : int, default=None
            Expected number of input samples, used to pre-compile the `__call__` method via a warm-up call.
            If `n_input` is `None`, the warm-up is skipped and `n_input` is set upon first use of `__call__`.
        memory_limit : float, optional
            Maximum memory in gigabytes to use during batched evaluation. Controls the batch size
            to stay within this memory limit. Default is 4.0.
        """
        self.__d_in = len(k)
        self.__d_out = d_out

        self.__node_gen = node_gen
        self.__is_nested = node_gen.is_nested

        self.__k = k
        self.__t = t

        self.__offset = 0
        self.__n_2_F = {}
        self.__n_2_nodes = {}
        self.__n_2_weights = {}
        self.__n_2_sorted_dims = {}
        self.__n_2_sorted_degs = {}
        self.__n_2_zetas = {}

        self.__n_f_evals = indices.nodeset_cardinality(k, t, nested=self.__is_nested)
        self.__n_f_evals_new = 0

        self.__compiled_tensor_product_evaluation = None
        self.__compiled_tensor_product_gradient = {}

        self.__memory_limit = memory_limit
        self.__n_inputs = n_inputs

        if f is not None:
            self.set_f(f=f)

    def set_f(
        self,
        *,
        f: Callable[[Union[jax.Array, np.ndarray]], Union[jax.Array, np.ndarray]],
        f_evals: dict[tuple, dict[tuple, jax.Array]] = None,
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

        Returns
        -------
        dict
            An updated dictionary containing all computed evaluations of the target function `f`.
        """
        if f_evals is None:
            f_evals = {}

        # Caching the interpolation node for nu = (0,0,...,0) for reuse in self.set_f
        zero = np.array([g(0)[0] for g in self.__node_gen])

        # get multi-indices and smolyak coefficients (only non-zero), binned by the number of active dimensions n
        n_2_nus, self.__n_2_zetas = indices.non_zero_indices_and_zetas(self.__k, self.__t)

        for n, nus in n_2_nus.items():

            # ----- Store constant term (offset) of the Smolyak operator -----

            if n == 0:
                assert len(self.__n_2_zetas[n]) == 1

                nu = ()
                if self.__is_nested:
                    f_evals_nu = f_evals
                else:
                    f_evals_nu = f_evals.get(nu, {})
                if nu not in f_evals_nu.keys():
                    f_evals_nu[nu] = f(zero.copy())
                    self.__n_f_evals_new += 1
                if not self.__is_nested:
                    f_evals[nu] = f_evals_nu

                self.__offset = self.__n_2_zetas[n][0] * f_evals_nu[nu]
                continue

            # ----- Assemble sorted dimensions and sorted degrees -----

            nn = len(nus)  # number of multi-indices of length n
            sorted_dims = np.empty((nn, n), dtype=int)
            sorted_degs = np.empty((nn, n), dtype=int)

            for i, nu in enumerate(n_2_nus[n]):
                sorted_nu = sorted(nu, key=lambda x: x[1], reverse=True)
                sorted_dims[i], sorted_degs[i] = zip(*sorted_nu)

            # ----- Assemble nodes and weights -----

            tau = tuple(int(ti) for ti in sorted_degs.max(axis=0))  # per-dimension maximal degree tau_i
            nodes = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]
            weights = [np.zeros((nn, tau_i + 1), dtype=float) for tau_i in tau]

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
                    nodes[t][idxs, :L] = pts
                    weights[t][idxs, :L] = wts

                # we can do even better if we vectorize node_gen(degrees) for isotropic rules, like GH or Leja

            # ----- Assemble the array storing the functions evaluations -----

            F = np.zeros((nn, self.__d_out) + tuple(ti + 1 for ti in tau), dtype=float)

            for i, nu in enumerate(n_2_nus[n]):
                x = zero.copy()

                if self.__is_nested:
                    f_evals_nu = f_evals
                else:
                    f_evals_nu = f_evals.get(nu, {})

                F_i = F[i]
                s_i = sorted_dims[i]
                argsort_s_i = np.argsort(s_i)

                ranges = [range(k + 1) for k in sorted_degs[i]]
                for mu_degrees in it.product(*ranges):
                    mu_tuple = tuple((s_i[i], mu_degrees[i]) for i in argsort_s_i if mu_degrees[i] > 0)
                    if mu_tuple not in f_evals_nu:
                        x[s_i] = [xi_k[i][deg] for xi_k, deg in zip(nodes, mu_degrees)]
                        f_evals_nu[mu_tuple] = f(x)
                        self.__n_f_evals_new += 1
                    F_i[:, *mu_degrees] = f_evals_nu[mu_tuple]

                if not self.__is_nested:
                    f_evals[nu] = f_evals_nu

            # save as jnp data structures
            self.__n_2_F[n] = jnp.array(F)
            self.__n_2_sorted_degs[n] = jnp.array(sorted_degs)
            self.__n_2_sorted_dims[n] = jnp.array(sorted_dims)
            self.__n_2_nodes[n] = [jnp.array(xi) for xi in nodes]
            self.__n_2_weights[n] = [jnp.array(w) for w in weights]
            self.__n_2_zetas[n] = jnp.array(self.__n_2_zetas[n])

        self.__setup_eval_functions()

        return f_evals

    def __setup_eval_functions(self) -> None:

        def __create_evaluate_tensor_product_gradient(n: int):
            def __evaluate_tensor_product_gradient_wrapped(x, F, *args):
                xi_list = args[:n]
                w_list = args[n : 2 * n]
                s_list = args[2 * n]
                nu = args[2 * n + 1]
                zeta = args[2 * n + 2]
                return barycentric.evaluate_tensor_product_gradient(x, F, xi_list, w_list, s_list, nu, zeta)

            return jax.vmap(jax.jit(__evaluate_tensor_product_gradient_wrapped), in_axes=(None, 0) + (0,) * (2 * n + 3))

        for n in self.__n_2_F.keys():
            self.__compiled_tensor_product_gradient[n] = __create_evaluate_tensor_product_gradient(n)

        self.__compiled_tensor_product_gradient = barycentric.evaluate_tensor_product_gradient
        self.__compiled_tensor_product_evaluation = barycentric.evaluate_tensor_product_interpolant

        if self.__n_inputs is not None:
            _ = self(jax.random.uniform(jax.random.PRNGKey(0), (self.__n_inputs, self.__d_in)))

    def __validate_input(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        assert bool(self.__compiled_tensor_product_evaluation) == bool(
            self.__n_2_F
        ), "The operator has not yet been set up for a target function via `set_f`."
        x = jnp.asarray(x)
        if x.shape == (self.__d_in,):
            x = x[None, :]
        assert x.shape[1] == self.__d_in, f"{x.shape[1]} != {self.__d_in}"
        self.__n_inputs = x.shape[0]
        return x

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
        x = self.__validate_input(x)
        I_Lambda_x = jnp.broadcast_to(self.__offset, (self.__n_inputs, self.__d_out))

        for n in self.__n_2_F.keys():

            n_summands = self.__n_2_F[n].shape[0]
            memory_per_summand = (
                self.__n_inputs**1.5 * self.__d_out * np.prod(self.__n_2_F[n].shape[3:]) * 8 / (1024**3)
            )
            summands_batch_size = max(1, int(np.floor(self.__memory_limit / memory_per_summand)))

            # batched processing of tensor product interpolants with n active dimensions
            for start_s in range(0, n_summands, summands_batch_size):

                end_s = min(start_s + summands_batch_size, n_summands)
                res = self.__compiled_tensor_product_evaluation(
                    x,
                    self.__n_2_F[n][start_s:end_s],
                    [arr[start_s:end_s] for arr in self.__n_2_nodes[n]],
                    [arr[start_s:end_s] for arr in self.__n_2_weights[n]],
                    self.__n_2_sorted_dims[n][start_s:end_s],
                    self.__n_2_sorted_degs[n][start_s:end_s],
                    self.__n_2_zetas[n][start_s:end_s],
                )
                I_Lambda_x += res

        return I_Lambda_x

    def gradient(self, x: Union[jax.Array, np.ndarray]) -> jax.Array:
        """
        Compute the gradient of the Smolyak interpolant at the given points.

        Parameters
        ----------
        x : Union[jax.Array, numpy.ndarray]
            Points at which to evaluate the gradient. Shape: `(n_points, d_in)`.

        Returns
        -------
        jax.Array
            Gradient of the interpolant evaluated at `x`.
            Shape: `(n_points, d_out, d_in)`.
        """
        x = self.__validate_input(x)
        J_Lambda_x = jnp.zeros((x.shape[0], self.__d_out, self.__d_in))

        for n in self.__n_2_F.keys():

            # determine the number of batches that ensures that the computation stays within the given memory limit
            n_summands = self.__n_2_F[n].shape[0]
            memory_per_summand = J_Lambda_x.size * np.prod(self.__n_2_F[n].shape[3:]) * 8 / (1024**3)
            summands_batch_size = max(1, int(np.floor(self.__memory_limit / memory_per_summand)))

            # batched processing of tensor product gradients with n active dimensions
            for start_s in range(0, n_summands, summands_batch_size):
                end_s = min(start_s + summands_batch_size, n_summands)
                res = self.__compiled_tensor_product_gradient(
                    x,
                    self.__n_2_F[n][start_s:end_s],
                    [arr[start_s:end_s] for arr in self.__n_2_nodes[n]],
                    [arr[start_s:end_s] for arr in self.__n_2_weights[n]],
                    self.__n_2_sorted_dims[n][start_s:end_s],
                    self.__n_2_sorted_degs[n][start_s:end_s],
                    self.__n_2_zetas[n][start_s:end_s],
                )
                J_Lambda_x += res  # jnp.sum(res, axis=0)

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
        # assemble quadrature weights, closely following the logic in set_f to assemble nodes and weights
        # ----------------------------------------------------------------------------
        n_2_quad_weights = {}

        for n in self.__n_2_F.keys():

            nn = len(self.__n_2_zetas[n])  # number of multi-indices of length n
            sorted_degs = self.__n_2_sorted_degs[n]
            sorted_dims = self.__n_2_sorted_dims[n]
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
        Q_Lambda = jnp.broadcast_to(self.__offset, self.__d_out)
        for n in self.__n_2_F.keys():
            q_id = "z"
            w_ids = string.ascii_lowercase[:n]
            o_id = string.ascii_lowercase[n]
            subscripts = f"{q_id}," + ",".join(f"{q_id}{w}" for w in w_ids) + f",{q_id}{o_id}{''.join(w_ids)}->{o_id}"
            Q_Lambda += jnp.einsum(subscripts, self.__n_2_zetas[n], *n_2_quad_weights[n], self.__n_2_F[n])
        return Q_Lambda.block_until_ready()
