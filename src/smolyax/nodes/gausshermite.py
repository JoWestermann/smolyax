from functools import lru_cache
from typing import Union

import jax
import numpy as np

from .base import Generator, Generator1D


class GaussHermite1D(Generator1D):
    """
    Generator for one-dimensional Gauss-Hermite nodes.

    Provides Gauss-Hermite quadrature nodes and weights on the real line, with optional scaling and shifting to custom
    domains.
    """

    def __init__(self, mean: float = 0.0, scaling: float = 1.0) -> None:
        """
        Initialize the one-dimensional Gauss-Hermite node generator.

        Parameters
        ----------
        mean : float, optional
            Center of the node sequence. Default is 0.0.
        scaling : float, optional
            Scaling of the node sequence. Default is 1.0.
        """
        super().__init__(is_nested=False)
        self.__mean = mean
        self.__scaling = scaling
        self.__cached_call = self.__make_cached_call()
        self.__cached_get_quadrature_weights = self.__make_cached_get_quadrature_weights()

    @property
    def mean(self) -> float:
        """
        Center of the node sequence.

        Returns
        -------
        float
        """
        return self.__mean

    @property
    def scaling(self) -> float:
        """
        Scaling factor of the node sequence.

        Returns
        -------
        float
        """
        return self.__scaling

    def __make_cached_call(self):
        @lru_cache(maxsize=None)
        def cached(n):
            return self.scale(np.polynomial.hermite.hermgauss(n + 1)[0])

        return cached

    def __call__(self, n: int) -> Union[jax.Array, np.ndarray]:
        """@public
        Generate the Gauss-Hermite node sequence of length `n+1`, scaled and shifted according to `self.scaling` and
        `self.mean`.

        Parameters
        ----------
        n : int
            Maximal node index (counting starts at `0`).

        Returns
        -------
        jax.Array or np.ndarray
            Scaled Gauss-Hermite nodes.
        """
        return self.__cached_call(n)

    @staticmethod
    def __make_cached_get_quadrature_weights():
        @lru_cache(maxsize=None)
        def cached(n):
            return np.polynomial.hermite.hermgauss(n + 1)[1] / np.sqrt(np.pi)

        return cached

    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Map nodes or points from the reference domain (corresponding to `mean=0` and `scaling=1`) to the custom domain.

        Parameters
        ----------
        x : jax.Array or np.ndarray
            Reference domain nodes or points.

        Mainly intended for internal use and testing.

        Returns
        -------
        jax.Array or np.ndarray
            Scaled nodes or points.
        """
        return self.__mean + self.__scaling * x

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Map nodes or points from the custom domain back to the reference domain (corresponding to `mean=0` and
        `scaling=1`) .

        Mainly intended for internal use and testing.

        Parameters
        ----------
        x : jax.Array or np.ndarray
            Nodes or points in the custom domain.

        Returns
        -------
        jax.Array or np.ndarray
            Nodes or points in the reference domain.
        """
        return (x - self.__mean) / self.__scaling

    def get_random(self, n: int = 1) -> Union[jax.Array, np.ndarray]:
        r"""
        Sample random points $x \sim \mathcal{N}(\mu, \frac{\sigma}{2})$, where $\mu$ = `self.mean` and
        $\sigma$ = `self.scaling`.

        The factor $\frac{1}{2}$ ensures that for $\mu=0$ and $\sigma=1$ we sample from the Gaussian measure with
        density $f(x) = \frac{1}{\sqrt{\pi}} exp(-x^2)$, which is the measure of integration for classical
        Gauss-Hermite quadrature.

        Parameters
        ----------
        n : int, optional
            Number of random points to sample (default is 1).

        Returns
        -------
        jax.Array or np.ndarray
            Randomly sampled points.
        """
        return self.scale(np.random.randn(n) / np.sqrt(2))

    def get_quadrature_weights(self, n: int) -> Union[jax.Array, np.ndarray]:
        """
        Return quadrature weights for Gauss-Hermite nodes.

        Parameters
        ----------
        n : int
            Maximal node index (counting starts at `0`).

        Returns
        -------
        jax.Array or np.ndarray
            Weights corresponding to the first `n+1` nodes.
        """
        return self.__cached_get_quadrature_weights(n)

    def __repr__(self) -> str:
        return f"Gauss-Hermite (mean = {self.__mean}, scaling = {self.__scaling})"


class GaussHermite(Generator):
    """
    Container for multiple 1D Gauss-Hermite node sequences generators with optional scaling and shifting in each
    dimension.
    """

    def __init__(
        self, mean: Union[jax.Array, np.ndarray] = None, scaling: Union[jax.Array, np.ndarray] = None, dim: int = None
    ):
        """
        Initialize the multidimensional Gauss-Hermite node generator.

        Parameters
        ----------
        mean : Union[jax.Array, np.ndarray], optional
            Node sequence centers for each dimension. Defaults to zeros.
        scaling : Union[jax.Array, np.ndarray], optional
            Node sequence scalings for each dimension. Defaults to ones.
        dim : int, optional
            Number of dimensions. Only required if neither `mean` nor `scaling` is provided.

        Raises
        ------
        ValueError
            If the dimension cannot be inferred from inputs.
        """
        if dim is None:
            if scaling is not None:
                dim = len(scaling)
            elif mean is not None:
                dim = len(mean)
            else:
                raise ValueError("Must specify at least one of 'dim', 'mean', or 'scaling'.")

        if mean is None and scaling is None:
            Generator.__init__(self, [GaussHermite1D()] * dim)
            self.__mean = np.zeros(dim)
            self.__scaling = np.ones(dim)
        else:
            self.__mean = np.zeros(dim) if mean is None else np.asarray(mean)
            self.__scaling = np.ones(dim) if scaling is None else np.asarray(scaling)
            Generator.__init__(self, [GaussHermite1D(m, a) for m, a in zip(self.__mean, self.__scaling)])

    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Map multidimensional points from the reference domain (corresponding to `mean=0` and `scaling=1` in each
        dimension) to the custom domain.

        Mainly intended for internal use and testing.

        Parameters
        ----------
        x : jax.Array or np.ndarray
            Nodes or points in reference domain.

        Returns
        -------
        jax.Array or np.ndarray
            Nodes or points in custom domain.
        """
        return self.__mean + self.__scaling * x

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Map multidimensional points from the custom domain back to the reference domain (corresponding to `mean=0` and
        `scaling=1` in each dimension).

        Mainly intended for internal use and testing.

        Parameters
        ----------
        x : jax.Array or np.ndarray
            Nodes or points in custom domain.

        Returns
        -------
        jax.Array or np.ndarray
            Nodes or points in reference domain.
        """
        return (x - self.__mean) / self.__scaling

    def __repr__(self) -> str:
        return f"Gauss Hermite (d = {self.dim}, mean = {self.__mean.tolist()}, scaling = {self.__scaling.tolist()})"
