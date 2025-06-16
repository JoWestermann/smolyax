from abc import ABC, abstractmethod
from typing import Iterator, List, Union

import jax
import numpy as np


class Generator1D(ABC):
    """
    Abstract base class for one-dimensional node sequences.

    These sequences can be used for interpolation or quadrature. Nodes are typically
    defined on a reference domain, but `scale` and `scale_back` allow mapping to custom domains.
    Supports random sampling from the measure of integration associated with the domain.
    """

    def __init__(self, is_nested: bool) -> None:
        """
        Initialize the node sequence generator.

        Parameters
        ----------
        is_nested : bool
            Whether the node sequence is nested.
        """
        self.__is_nested = is_nested

    @property
    def is_nested(self) -> bool:
        """
        Whether the node sequence is nested.

        Returns
        -------
        bool
        """
        return self.__is_nested

    @abstractmethod
    def __call__(self, n: int) -> Union[jax.Array, np.ndarray]:
        """@public
        Generate a node sequence of length `n+1`.

        Parameters
        ----------
        n : int
            Maximal node index (counting starts at `0`).

        Returns
        -------
        jax.Array or np.ndarray
            The generated node sequence.
        """
        ...

    @abstractmethod
    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Map nodes and points from the reference domain to a custom domain.

        Mainly intended for internal use and testing.

        Parameters
        ----------
        x : jax.Array or np.ndarray
            Nodes or points on the reference domain.

        Returns
        -------
        jax.Array or np.ndarray
            Nodes or points mapped to the custom domain.
        """
        ...

    @abstractmethod
    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Map nodes and points from the custom domain back to the reference domain.

        Mainly intended for internal use and testing.

        Parameters
        ----------
        x : jax.Array or np.ndarray
            Nodes or points on the custom domain.

        Returns
        -------
        jax.Array or np.ndarray
            Nodes or points mapped to the reference domain.
        """
        ...

    @abstractmethod
    def get_random(self, n: int = 1) -> Union[jax.Array, np.ndarray]:
        """
        Sample points randomly from the measure associated with the domain.

        This measure corresponds, for example, to the integration measure for quadrature.

        Parameters
        ----------
        n : int, optional
            Number of random points to sample (default is 1).

        Returns
        -------
        jax.Array or np.ndarray
            Randomly sampled points.
        """
        ...

    @abstractmethod
    def get_quadrature_weights(self, n: int) -> Union[jax.Array, np.ndarray]:
        """
        Return the quadrature weights for the node sequence.

        Parameters
        ----------
        n : int
            Weights corresponding to the first `n+1` nodes.

        Returns
        -------
        jax.Array or np.ndarray
            Quadrature weights.
        """
        ...


class Generator:
    """
    Container for multiple 1D node sequences generators, to be used for creating multi-dimensional sparse grids.

    Supports scaling to custom domains and random sampling.
    """

    def __init__(self, node_gens: List[Generator1D]):
        """
        Initialize the generator.

        Parameters
        ----------
        node_gens : list of Generator1D
            List of 1D node sequence generators.
        """
        assert all(g.is_nested == node_gens[0].is_nested for g in node_gens)
        self.__is_nested = node_gens[0].is_nested
        self.__dim = len(node_gens)
        self.__gens = node_gens

    @property
    def dim(self) -> int:
        """
        Dimensionality of the generator.

        Returns
        -------
        int
            Number of dimensions.
        """
        return self.__dim

    @property
    def is_nested(self) -> bool:
        """
        Whether all node sequences are nested.

        Returns
        -------
        bool
        """
        return self.__is_nested

    def __getitem__(self, i: int) -> Generator1D:
        """
        Get the 1D node generator associated with dimension `i`.

        Parameters
        ----------
        i : int
            Dimension index of the 1D generator.

        Returns
        -------
        Generator1D
        """
        return self.__gens[i]

    def __iter__(self) -> Iterator[Generator1D]:
        """
        Iterate over the 1D node generators.

        Returns
        -------
        iterator of Generator1D
        """
        for index in range(self.__dim):
            yield self[index]

    def get_random(self, n: int = 0) -> Union[jax.Array, np.ndarray]:
        """
        Generate random points from the multi-dimensional measure.

        Parameters
        ----------
        n : int, optional
            Number of random points (default is 0, meaning one sample per dimension).

        Returns
        -------
        jax.Array or np.ndarray
            Random points.
        """
        if n == 0:
            return np.squeeze([g.get_random() for g in self.__gens])
        return np.array([g.get_random(n) for g in self.__gens]).T

    def scale(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Scale multi-dimensional nodes or points from reference to custom domain.

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
        assert x.shape[-1] == self.dim
        if x.ndim == 1:
            return np.array([g.scale(xi) for g, xi in zip(self.__gens, x)])
        else:
            return np.array([g.scale(xi) for g, xi in zip(self.__gens, x.T)]).T

    def scale_back(self, x: Union[jax.Array, np.ndarray]) -> Union[jax.Array, np.ndarray]:
        """
        Scale multi-dimensional nodes or points from custom domain back to reference domain.

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
        assert x.shape[-1] == self.dim
        if x.ndim == 1:
            return np.array([g.scale_back(xi) for g, xi in zip(self.__gens, x)])
        else:
            return np.array([g.scale_back(xi) for g, xi in zip(self.__gens, x.T)]).T
