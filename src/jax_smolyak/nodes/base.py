from abc import ABC, abstractmethod
from typing import Iterator, List, Self

import numpy as np
from numpy.typing import ArrayLike


class Generator(ABC):
    """Abstract base class for univariate interpolation points"""

    def __init__(self, dim: int, is_nested: bool) -> None:
        self._dim = dim
        self._is_nested = is_nested

    @property
    def is_nested(self) -> bool:
        return self._is_nested

    @property
    def dim(self) -> int:
        return self._dim

    @abstractmethod
    def __call__(self, n: int) -> ArrayLike: ...

    def __getitem__(self, i: int):
        assert i == 0
        return self

    def __iter__(self) -> Iterator[Self]:
        for index in range(self.dim):
            yield self[index]

    @abstractmethod
    def scale(self, x: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def scale_back(self, x: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def get_random(self, n: int = 1) -> ArrayLike: ...

    def get_zero(self) -> ArrayLike:
        return self(0)[0]

    @abstractmethod
    def __repr__(self) -> str: ...


class GeneratorMultiD(Generator):
    """Abstract base class for multivariate interpolation points"""

    def __init__(self, node_gens: List[Generator]):
        assert all(g.is_nested == node_gens[0].is_nested for g in node_gens)
        super().__init__(dim=len(node_gens), is_nested=node_gens[0].is_nested)
        self.gens = node_gens

    def __call__(self, n: int) -> ArrayLike:
        raise

    def __getitem__(self, i: int) -> Generator:
        return self.gens[i]

    def get_zero(self) -> ArrayLike:
        return np.array([g(0)[0] for g in self.gens])

    def get_random(self, n: int = 0) -> ArrayLike:
        if n == 0:
            return np.squeeze([g.get_random() for g in self.gens])
        return np.array([g.get_random(n) for g in self.gens]).T

    def scale(self, x: ArrayLike) -> ArrayLike:
        assert x.shape[-1] == self.dim
        if x.ndim == 1:
            return np.array([g.scale(xi) for g, xi in zip(self.gens, x)])
        else:
            return np.array([g.scale(xi) for g, xi in zip(self.gens, x.T)]).T

    def scale_back(self, x: ArrayLike) -> ArrayLike:
        assert x.shape[-1] == self.dim
        if x.ndim == 1:
            return np.array([g.scale_back(xi) for g, xi in zip(self.gens, x)])
        else:
            return np.array([g.scale_back(xi) for g, xi in zip(self.gens, x.T)]).T

    @abstractmethod
    def __repr__(self) -> str: ...
