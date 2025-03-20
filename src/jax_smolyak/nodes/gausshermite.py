import numpy as np
from numpy.typing import ArrayLike

from .base import Generator, GeneratorMultiD


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

    def __init__(
        self, mlist: ArrayLike = None, alist: ArrayLike = None, dim: int = None
    ):
        dim = dim
        if dim is None:
            if alist is not None:
                dim = len(alist)
            elif mlist is not None:
                dim = len(mlist)
            else:
                raise

        if mlist is None:
            mlist = np.zeros(dim)
        if alist is None:
            alist = np.ones(dim)

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

    def scale(self, x: ArrayLike) -> ArrayLike:
        return self.mean + self.scaling * x

    def scale_back(self, x: ArrayLike) -> ArrayLike:
        return (x - self.mean) / self.scaling
