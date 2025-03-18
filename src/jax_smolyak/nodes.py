from abc import ABC, abstractmethod
from typing import Iterator, List, Self

import numpy as np
from numpy.typing import ArrayLike

# =========== Abstract Base Classes ===========


class Generator(ABC):
    """Abstract base class for interpolation points"""

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


# =========== Gauss-Hermite nodes ===========


class GaussHermite1D(Generator):
    """Gauss-Hermite grid points"""

    def __init__(self, mean: float, scaling: float) -> None:
        super().__init__(dim=1, is_nested=False)
        self.mean = mean
        self.scaling = scaling

    def __repr__(self) -> str:
        return f"Gauss-Hermite (mean = {self.mean}, scaling = {self.scaling})"

    def __call__(self, n: int) -> ArrayLike:
        nodes = np.polynomial.hermite.hermgauss(n + 1)[0]
        return self.scale(nodes)

    def scale(self, x: ArrayLike) -> ArrayLike:
        return self.mean + self.scaling * x

    def scale_back(self, x: ArrayLike) -> ArrayLike:
        return (x - self.mean) / self.scaling

    def get_random(self, n: int = 1) -> ArrayLike:
        return self.scale(np.random.randn(n))


class GaussHermite(GeneratorMultiD):

    def __init__(self, mlist: ArrayLike, alist: ArrayLike):
        GeneratorMultiD.__init__(
            self, [GaussHermite1D(m, a) for m, a in zip(mlist, alist)]
        )
        self.mean = np.array(mlist)
        self.scaling = np.array(alist)

    def __repr__(self) -> str:
        return (
            f"Gauss Hermite (d = {self.dim}"
            f", mean = {self.mean.tolist()}"
            f", scaling = {self.scaling.tolist()})"
        )


# =========== Leja nodes ===========


class Leja1D(Generator):
    """Leja grid points"""

    nodes = np.array([0, 1, -1, 1 / np.sqrt(2), -1 / np.sqrt(2)])
    _reference_domain = (-1, 1)

    def __init__(self, domain: ArrayLike) -> None:
        super().__init__(dim=1, is_nested=True)
        if domain is None:
            domain = self._reference_domain
        self.domain = domain

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

    def __call__(self, n: int) -> ArrayLike:
        self._ensure_nodes(n)
        if self.domain is None:
            return self.nodes[: n + 1]
        return self.scale(self.nodes[: n + 1], self._reference_domain, self.domain)

    def scale(
        self, x: ArrayLike, d1: ArrayLike = None, d2: ArrayLike = None
    ) -> ArrayLike:
        """
        Affine transformation from the interval d1 to the interval d2 applied to point x.
        x : scalar or array or list of shape (n, d) or (d, )
        d1, d2 : arrays or lists of shape (d, 2) or (2, )
        """
        if d1 is None:
            d1 = self._reference_domain
        if d2 is None:
            d2 = self.domain

        # ensure d1, d2 have shape (d, 2)
        d1, d2 = np.array(d1), np.array(d2)
        assert d1.shape == d2.shape
        d = 1 if len(d1.shape) == 1 else d1.shape[0]
        if len(d1.shape) == 1:
            d1 = d1.reshape((1, 2))
            d2 = d2.reshape((1, 2))
        for i in range(d):
            assert d1[i, 0] < d1[i, 1]
            assert d2[i, 0] < d2[i, 1]

        # ensure x has shape (n, d)
        x = np.array(x)
        x_shape = x.shape
        if x_shape == ():
            x = np.array([[x]])
        else:
            x = np.array(x)
            if len(x.shape) == 1:
                if x.shape[0] == d:
                    x = x.reshape((1, len(x)))
                else:
                    x = x.reshape((len(x), 1))

        # ensure x in d1
        for i in range(d):
            for j in range(x.shape[0]):
                assert x[j, i] >= d1[i, 0] or np.isclose(
                    x[j, i], d1[i, 0]
                ), f"Assertion failed with\n x[{j},{i}] ({x[j, i]})\n d1[{i},0] ({d1[i, 0]})"
                assert x[j, i] <= d1[i, 1] or np.isclose(
                    x[j, i], d1[i, 1]
                ), f"Assertion failed with\n x[{j},{i}] ({x[j, i]})\n d1[{i},1] ({d1[i, 1]})"

        # check
        assert len(x.shape) == len(d1.shape) == len(d2.shape) == 2
        assert x.shape[1] == d1.shape[0] == d2.shape[0]
        assert d1.shape[1] == d2.shape[1] == 2

        # scale
        for i in range(d):
            x[:, i] = (x[:, i] - d1[i, 0]) / (d1[i, 1] - d1[i, 0])
            x[:, i] = x[:, i] * (d2[i, 1] - d2[i, 0]) + d2[i, 0]

        # Return in original shape
        return x.reshape(x_shape)

    def scale_back(self, x: ArrayLike) -> ArrayLike:
        return self.scale(x, d1=self.domain, d2=self._reference_domain)

    def get_random(self, n: int = 1):
        if self.domain is None:
            return np.random.uniform(-1, 1, n)
        return np.random.uniform(self.domain[0], self.domain[1], n)


class Leja(GeneratorMultiD):

    def __init__(self, *, domains: ArrayLike = None, d: int = None):
        if domains is not None:
            GeneratorMultiD.__init__(self, [Leja1D(domain) for domain in domains])
        elif d is not None:
            GeneratorMultiD.__init__(self, [Leja1D(domain=(-1, 1)) for _ in range(d)])
        else:
            raise
        self.domains = np.array(domains)

    def __repr__(self) -> str:
        return f"Leja (d = {self.dim}, domains = {self.domains.tolist()})"
