from numpy.typing import ArrayLike

from .base import Generator, GeneratorMultiD


class MyNodes1D(Generator):

    def __init__(self) -> None:
        super().__init__(dim=1, is_nested=False)

    def __repr__(self) -> str:
        return f"MyNodes1D"

    def __call__(self, n: int) -> ArrayLike:
        raise NotImplementedError

    def scale(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def scale_back(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def get_random(self, n: int = 1) -> ArrayLike:
        raise NotImplementedError


class MyNodes(GeneratorMultiD):

    def __init__(self, d: int):
        GeneratorMultiD.__init__(self, [MyNodes1D() for _ in range(d)])

    def __repr__(self) -> str:
        return "MyNodes"
