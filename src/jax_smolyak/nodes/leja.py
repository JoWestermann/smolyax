from functools import lru_cache
from typing import Sequence, Union

import jax
import numpy as np

from .base import Generator, GeneratorMultiD


class Leja1D(Generator):
    """Leja grid points"""

    nodes = np.array([0, 1, -1, 1 / np.sqrt(2), -1 / np.sqrt(2)])

    def __init__(self, domain: Union[jax.Array, np.ndarray, Sequence[float]] = None) -> None:
        super().__init__(dim=1, is_nested=True)
        self.domain = domain
        self._reference_domain = None
        if domain is not None:
            self.domain = np.asarray(domain)
            self._reference_domain = (-1, 1)
        self.__cached_scaled_call = self.__make_cached_scaled_call()

    def __repr__(self) -> str:
        return f"Leja (domain = {self.domain})"

    @classmethod
    def _ensure_nodes(cls, n: int):
        k = cls.nodes.shape[0]
        if n >= k:
            cls.nodes = np.append(cls.nodes, np.empty((n + 1 - k,)))
            for j in range(k, n + 1):
                if j % 2 == 0:
                    cls.nodes[j] = -cls.nodes[j - 1]
                else:
                    cls.nodes[j] = np.sqrt((cls.nodes[int((j + 1) / 2)] + 1) / 2)

    def __make_cached_scaled_call(self):
        @lru_cache(maxsize=None)
        def cached(n):
            self._ensure_nodes(n)
            return self.scale(self.nodes[: n + 1])

        return cached

    def __call__(self, n: int) -> Union[jax.Array, np.ndarray]:
        return self.__cached_scaled_call(n)

    def scale(
        self,
        x: Union[jax.Array, np.ndarray],
        d1: Union[jax.Array, np.ndarray, Sequence[float]] = None,
        d2: Union[jax.Array, np.ndarray, Sequence[float]] = None,
    ) -> Union[jax.Array, np.ndarray]:
        """
        Affine transformation from the interval d1 to the interval d2 applied to point x.
        x : scalar or array or list of shape (n, )
        d1, d2 : arrays or lists of shape (2, )
        """
        if d1 is None:
            d1 = self._reference_domain
        if d2 is None:
            d2 = self.domain

        assert (d1 is None) == (d2 is None)
        if d1 is None:  # no scaling if no custom domains are give
            return x

        # ensure d1, d2 have shape (2, )
        d1, d2 = np.squeeze(d1), np.squeeze(d2)
        assert d1.shape == d2.shape == (2,), f"shapes {d1.shape} and {d2.shape} do not match (2, )"
        assert d1[0] < d1[1]
        assert d2[0] < d2[1]

        # ensure x has shape (n, )
        x_shape = x.shape
        x = np.squeeze(x)
        assert x.ndim <= 1

        # ensure x in d1
        valid_lower = (x >= d1[0]) | np.isclose(x, d1[0])
        valid_upper = (x <= d1[1]) | np.isclose(x, d1[1])
        assert np.all(valid_lower), f"Assertion failed: Some values are below lower bounds\n{x[~valid_lower]}"
        assert np.all(valid_upper), f"Assertion failed: Some values are above upper bounds\n{x[~valid_upper]}"

        # scale
        x = (x - d1[0]) / (d1[1] - d1[0])
        x = x * (d2[1] - d2[0]) + d2[0]

        # Return in original shape
        return x.reshape(x_shape)

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        return self.scale(x, d1=self.domain, d2=self._reference_domain)

    def get_random(self, n: int = 1):
        return self.scale(np.random.uniform(-1, 1, n))


class Leja(GeneratorMultiD):

    def __init__(
        self,
        *,
        domains: list[Union[jax.Array, np.ndarray, Sequence[float]]] = None,
        dim: int = None,
    ):
        self.domains = None
        self._reference_domains = None
        if domains is not None:
            GeneratorMultiD.__init__(self, [Leja1D(domain) for domain in domains])
            self.domains = np.asarray(domains)
            self._reference_domains = np.array([[-1, 1]] * len(domains))
        elif dim is not None:
            GeneratorMultiD.__init__(self, [Leja1D() for _ in range(dim)])
        else:
            raise

    def __repr__(self) -> str:
        if self.domains is not None:
            return f"Leja (d = {self.dim}, domains = {self.domains.tolist()})"
        else:
            return f"Leja (d = {self.dim})"

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        return self.scale(x, d1=self.domains, d2=self._reference_domains)

    def scale(
        self,
        x: Union[jax.Array, np.ndarray],
        d1: Union[jax.Array, np.ndarray, Sequence[float]] = None,
        d2: Union[jax.Array, np.ndarray, Sequence[float]] = None,
    ) -> Union[jax.Array, np.ndarray]:
        """
        Affine transformation from the interval d1 to the interval d2 applied to point x.
        x : array or list of shape (n, d) or (d, )
        d1, d2 : arrays or lists of shape (d, 2)
        """
        if d1 is None:
            d1 = self._reference_domains
        if d2 is None:
            d2 = self.domains

        assert (d1 is None) == (d2 is None)
        if d1 is None:  # no scaling if no custom domains are given
            return x

        # ensure d1, d2 have shape (d, 2)
        d1, d2 = np.asarray(d1), np.asarray(d2)
        assert d1.shape == d2.shape, f"shapes {d1.shape} and {d2.shape} do not match"
        d = 1 if len(d1.shape) == 1 else d1.shape[0]
        if len(d1.shape) == 1:
            d1 = d1.reshape((1, 2))
            d2 = d2.reshape((1, 2))
        for i in range(d):
            assert d1[i, 0] < d1[i, 1]
            assert d2[i, 0] < d2[i, 1]

        # ensure x has shape (n, d)
        x = np.asarray(x)
        x_shape = x.shape
        if x_shape == ():
            x = np.array([[x]])
        else:
            if len(x.shape) == 1:
                if x.shape[0] == d:
                    x = x.reshape((1, len(x)))
                else:
                    x = x.reshape((len(x), 1))

        # ensure x in d1
        valid_lower = (x >= d1[:, 0]) | np.isclose(x, d1[:, 0])
        valid_upper = (x <= d1[:, 1]) | np.isclose(x, d1[:, 1])
        assert np.all(valid_lower), f"Assertion failed: Some values are below lower bounds\n{x[~valid_lower]}"
        assert np.all(valid_upper), f"Assertion failed: Some values are above upper bounds\n{x[~valid_upper]}"

        # check
        assert len(x.shape) == len(d1.shape) == len(d2.shape) == 2
        assert x.shape[1] == d1.shape[0] == d2.shape[0]
        assert d1.shape[1] == d2.shape[1] == 2

        # scale
        x = (x - d1[:, 0]) / (d1[:, 1] - d1[:, 0])
        x = x * (d2[:, 1] - d2[:, 0]) + d2[:, 0]

        # Return in original shape
        return x.reshape(x_shape)
