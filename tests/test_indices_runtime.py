import sys

import numpy as np
import pytest

from jax_smolyak import indices


@pytest.mark.parametrize(
    "d, t, m",
    [
        (10000, 5.3, 98),
        (10000, 7.78, 1000),
        (10000, 10.317, 10003),
    ],
)
def test_indexset_runtime(benchmark, d, t, m):
    limit = d + 100
    if sys.getrecursionlimit() < limit:
        sys.setrecursionlimit(limit)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    iset = benchmark(lambda: indices.indexset(k, t))
    assert len(iset) == m


@pytest.mark.parametrize(
    "d, m, nested",
    [
        (100, 1000, True),
        (100, 1000, False),
        (10000, 10000, True),
        (10000, 10000, False),
    ],
)
def test_find_threshold_nested_runtime(benchmark, d, m, nested):
    accuracy = 0.01 if nested else 0.1
    limit = d + 100
    if sys.getrecursionlimit() < limit:
        sys.setrecursionlimit(limit)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    t = benchmark(lambda: indices.find_suitable_t(k, m, nested=nested, accuracy=accuracy))
    assert np.abs(indices.cardinality(k, t, nested=nested) - m) / m < accuracy, f'd={d}, m={m}'


@pytest.mark.parametrize(
    "d, t",
    [
        (10, 6.01),
        (10, 15.416),
        (100, 7.943),
        (10000, 10.317),
    ],
)
def test_smolyak_coeffs_runtime(benchmark, d, t):
    limit = d + 100
    if sys.getrecursionlimit() < limit:
        sys.setrecursionlimit(limit)
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    iset = indices.indexset(k, t)
    benchmark(lambda: [indices.smolyak_coefficient_zeta(k, t, nu=nu) for nu in iset])
