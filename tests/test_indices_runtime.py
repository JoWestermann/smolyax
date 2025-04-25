import numpy as np
import pytest

from jax_smolyak import indices


@pytest.mark.parametrize(
    "d, t, m",
    [
        (10, 6, 96),
        (10, 10.11, 1000),
    ],
)
def test_indexset_runtime(benchmark, d, t, m):
    k = np.log([2 + i for i in range(d)]) / np.log(2)
    iset = benchmark(lambda: indices.indexset(k, t))
    assert len(iset) == m
