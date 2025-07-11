import numpy as np
import pytest

from smolyax import indices, nodes
from smolyax.interpolation import SmolyakBarycentricInterpolator


def setup_nodes(d: int, node_type: str, default_domain: bool) -> nodes.Generator:
    """
    Set up a `nodes.Generator` instance based on the given type and domain configuration.

    Parameters
    ----------
    d : int
        The dimension of the domain.
    node_type : str
        The type of nodes to set up. Must be one of {'leja', 'gauss'}.
    default_domain : bool
        If True, use the default domain for the specified node type.
        If False, generate a random domain.

    Returns
    -------
    nodes.Generator
        An instance of the specified node type with the configured domain.
    """
    if node_type == "leja":
        if default_domain:
            return nodes.Leja(dim=d)
        else:
            domain = np.sort(np.random.rand(d, 2), axis=1)
            return nodes.Leja(domains=domain)
    elif node_type == "gauss":
        if default_domain:
            return nodes.GaussHermite(dim=d)
        else:
            mean = np.random.randn(d)
            scaling = np.random.rand(d)
            return nodes.GaussHermite(mean, scaling)
    else:
        raise ValueError("node_type must be one of {'leja', 'gauss'}")


@pytest.mark.parametrize(
    "d, m, node_type, default_domain",
    [
        (100, 1000, "leja", True),
        (100, 1000, "leja", False),
        (100, 1000, "gauss", True),
        (100, 1000, "gauss", False),
        (10000, 10000, "leja", True),
        (10000, 10000, "leja", False),
        (10000, 10000, "gauss", True),
        (10000, 10000, "gauss", False),
    ],
)
def test_smolyak_constructor_runtime(benchmark, d, m, node_type, default_domain):
    node_gen = setup_nodes(d, node_type, default_domain)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    t = indices.find_approximate_threshold(k, m, node_gen.is_nested)
    benchmark(lambda: SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=10))


def target_f(x, theta, r):
    return 1 / (1 + theta * np.sum(x * (np.arange(x.shape[-1]) + 2) ** (-r), axis=-1))


@pytest.mark.parametrize(
    "d, m, node_type, default_domain",
    [
        (100, 1000, "leja", True),
        (100, 1000, "leja", False),
        (100, 1000, "gauss", True),
        (100, 1000, "gauss", False),
        (10000, 10000, "leja", True),
        (10000, 10000, "leja", False),
        (10000, 10000, "gauss", True),
        (10000, 10000, "gauss", False),
    ],
)
def test_smolyak_set_f_runtime(benchmark, d, m, node_type, default_domain):
    node_gen = setup_nodes(d, node_type, default_domain)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    t = indices.find_approximate_threshold(k, m, node_gen.is_nested)
    f = lambda x: target_f(x, 2.0, 2.0)
    benchmark(lambda: SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=1, f=f))


@pytest.mark.parametrize(
    "d, m, node_type, default_domain",
    [
        (100, 1000, "leja", True),
        (100, 1000, "leja", False),
        (100, 1000, "gauss", True),
        (100, 1000, "gauss", False),
        (10000, 10000, "leja", True),
        (10000, 10000, "leja", False),
        (10000, 10000, "gauss", True),
        (10000, 10000, "gauss", False),
    ],
)
def test_smolyak_eval(benchmark, d, m, node_type, default_domain):
    node_gen = setup_nodes(d, node_type, default_domain)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    t = indices.find_approximate_threshold(k, m, node_gen.is_nested)
    f = lambda x: target_f(x, 2.0, 2.0)
    n_inputs = 250
    smol = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=1, n_inputs=n_inputs, f=f)
    x = np.random.randn(n_inputs, d)
    benchmark(lambda: smol(x))
