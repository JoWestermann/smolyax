from abc import ABC, abstractmethod
from typing import Iterator, List, Self, Union

import jax
import numpy as np


class Generator(ABC):
    """Abstract base class for univariate interpolation points"""

    def __init__(self, dim: int, is_nested: bool) -> None:
        self.__dim = dim
        self.__is_nested = is_nested

    @property
    def is_nested(self) -> bool:
        return self.__is_nested

    @property
    def dim(self) -> int:
        return self.__dim

    @abstractmethod
    def __call__(self, n: int) -> Union[jax.Array, np.ndarray]: ...

    def __getitem__(self, i: int):
        assert i == 0
        return self

    def __iter__(self) -> Iterator[Self]:
        for index in range(self.__dim):
            yield self[index]

    @abstractmethod
    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]: ...

    @abstractmethod
    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]: ...

    @abstractmethod
    def get_random(self, n: int = 1) -> Union[jax.Array, np.ndarray]: ...

    def get_zero(self) -> Union[jax.Array, np.ndarray]:
        return self(0)[0]

    @abstractmethod
    def __repr__(self) -> str: ...


class GeneratorMultiD(Generator):
    """Abstract base class for multivariate interpolation points"""

    def __init__(self, node_gens: List[Generator]):
        assert all(g.is_nested == node_gens[0].is_nested for g in node_gens)
        super().__init__(dim=len(node_gens), is_nested=node_gens[0].is_nested)
        self.__gens = node_gens

    def __call__(self, n: int) -> Union[jax.Array, np.ndarray]:
        raise

    def __getitem__(self, i: int) -> Generator:
        return self.__gens[i]

    def get_zero(self) -> Union[jax.Array, np.ndarray]:
        return np.array([g(0)[0] for g in self.__gens])

    def get_random(self, n: int = 0) -> Union[jax.Array, np.ndarray]:
        if n == 0:
            return np.squeeze([g.get_random() for g in self.__gens])
        return np.array([g.get_random(n) for g in self.__gens]).T

    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        assert x.shape[-1] == self.dim
        if x.ndim == 1:
            return np.array([g.scale(xi) for g, xi in zip(self.__gens, x)])
        else:
            return np.array([g.scale(xi) for g, xi in zip(self.__gens, x.T)]).T

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        assert x.shape[-1] == self.dim
        if x.ndim == 1:
            return np.array([g.scale_back(xi) for g, xi in zip(self.__gens, x)])
        else:
            return np.array([g.scale_back(xi) for g, xi in zip(self.__gens, x.T)]).T

    @abstractmethod
    def __repr__(self) -> str: ...
