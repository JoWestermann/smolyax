import numpy as np
import pytest

from jax_smolyak import indices, nodes, smolyak


def setup_nodes(d, nested, centered):
    if nested:
        if centered:
            return nodes.Leja(dim=d)
        else:
            domain = np.zeros((d, 2))
            domain[:, 1] = np.sort(np.random.rand(d) * 10)
            domain[:, 0] = -domain[:, 1]
            return nodes.Leja(domains=domain)
    else:
        if centered:
            return nodes.GaussHermite(dim=d)
        else:
            mean = np.random.randn(d)
            scaling = np.random.rand(d)
            return nodes.GaussHermite(mean, scaling)


@pytest.mark.parametrize(
    "d, m, nested, centered",
    [
        (100, 1000, True, True),
        (100, 1000, True, False),
        (100, 1000, False, True),
        (100, 1000, False, False),
        (10000, 10000, True, True),
        (10000, 10000, False, False),
        (10000, 10000, False, True),
        (10000, 10000, False, False),
    ],
)
def d_test_smolyak_constructor_runtime(benchmark, d, m, nested, centered):
    node_gen = setup_nodes(d, nested, centered)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    t = indices.find_approximate_threshold(k, m)
    benchmark(lambda: smolyak.SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=10))


def target_f(x, theta, r):
    return 1 / (1 + theta * np.sum(x * (np.arange(x.shape[-1]) + 2) ** (-r), axis=-1))


@pytest.mark.parametrize(
    "d, m, nested, centered",
    [
        (100, 1000, True, True),
        (100, 1000, True, False),
        (100, 1000, False, True),
        (100, 1000, False, False),
        (10000, 10000, True, True),
        (10000, 10000, False, False),
        (10000, 10000, False, True),
        (10000, 10000, False, False),
    ],
)
def test_smolyak_set_f_runtime(benchmark, d, m, nested, centered):
    node_gen = setup_nodes(d, nested, centered)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    t = indices.find_approximate_threshold(k, m)
    f = lambda x: target_f(x, 2.0, 2.0)
    benchmark(lambda: smolyak.SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=1, f=f))
