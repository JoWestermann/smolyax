from functools import lru_cache
from typing import Union

import jax
import numpy as np

from .base import Generator, GeneratorMultiD


class GaussHermite1D(Generator):
    """Generator for Gauss-Hermite points, a non-nested node sequence on the real line."""

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def scaling(self) -> float:
        return self.__scaling

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
        super().__init__(dim=1, is_nested=False)
        self.__mean = mean
        self.__scaling = scaling
        self.__cached_scaled_gh_n_plus_1 = self.__make_cached_scaled_gauss_hermite_n_plus_1()

    def __repr__(self) -> str:
        return f"Gauss-Hermite (mean = {self.__mean}, scaling = {self.__scaling})"

    def __make_cached_scaled_gauss_hermite_n_plus_1(self):
        @lru_cache(maxsize=None)
        def cached(n):
            return self.scale(np.polynomial.hermite.hermgauss(n + 1)[0])

        return cached

    def __call__(self, n: int) -> Union[jax.Array, np.ndarray]:
        return self.__cached_scaled_gh_n_plus_1(n)

    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        return self.__mean + self.__scaling * x

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        return (x - self.__mean) / self.__scaling

    def get_random(self, n: int = 1) -> Union[jax.Array, np.ndarray]:
        return self.scale(np.random.randn(n) / np.sqrt(2))

    def get_quadrature_weights(self, n: int) -> Union[jax.Array, np.ndarray]:
        return np.polynomial.hermite.hermgauss(n + 1)[1] / np.sqrt(np.pi)


class GaussHermite(GeneratorMultiD):
    """Multidimensional Gauss-Hermite node generator."""

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
        dim = dim
        if dim is None:
            if scaling is not None:
                dim = len(scaling)
            elif mean is not None:
                dim = len(mean)
            else:
                raise ValueError("Must specify at least one of 'dim', 'mean', or 'scaling'.")

        if mean is None:
            mean = np.zeros(dim)
        if scaling is None:
            scaling = np.ones(dim)

        GeneratorMultiD.__init__(self, [GaussHermite1D(m, a) for m, a in zip(mean, scaling)])

        self.__mean = np.asarray(mean)
        self.__scaling = np.asarray(scaling)

    def __repr__(self) -> str:
        return f"Gauss Hermite (d = {self.dim}, mean = {self.__mean.tolist()}, scaling = {self.__scaling.tolist()})"

    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        return self.__mean + self.__scaling * x

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        return (x - self.__mean) / self.__scaling
