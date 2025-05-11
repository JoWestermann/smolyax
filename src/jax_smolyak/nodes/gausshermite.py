from functools import lru_cache
from typing import Union

import jax
import numpy as np

from .base import Generator, GeneratorMultiD


class GaussHermite1D(Generator):
    """Gauss-Hermite grid points"""

    def __init__(self, mean: float, scaling: float) -> None:
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
        return self.scale(np.random.randn(n))


class GaussHermite(GeneratorMultiD):

    def __init__(
        self, mean: Union[jax.Array, np.ndarray] = None, scaling: Union[jax.Array, np.ndarray] = None, dim: int = None
    ):
        dim = dim
        if dim is None:
            if scaling is not None:
                dim = len(scaling)
            elif mean is not None:
                dim = len(mean)
            else:
                raise

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
