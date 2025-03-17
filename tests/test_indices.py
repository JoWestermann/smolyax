import itertools as it

import numpy as np

from jax_smolyak.indices import *


def get_random_indexsets(nested=False):
    d = np.random.randint(low=1, high=6)

    a = 1.1 + 2.9 * np.random.random()  # a \in [1.1,4.)
    b = 0.1 + 1.9 * np.random.random()  # b \in [0.1,3)
    k = np.log([a + b * i for i in range(d)]) / np.log(a)

    n_t = np.random.randint(low=1, high=100)
    l = find_suitable_t(k, n_t, nested=nested)

    isparse = indexset_sparse(lambda j: k[j], l, cutoff=d)
    idense = indexset_dense(k, l)
    print(
        f"\tConstructed {d}-dimensional multi-index sets with a={a}, b={b} and target n={n_t}. "
        + f"Sets are of cardinality {len(isparse)}."
    )
    return k, l, isparse, idense


def test_validity_of_indexsets():
    print("Testing that index sets contain the correct multi-indices and none extra.")

    for i in range(10):
        k, l, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for idx_dense in it.product(*[range(int(np.floor(ki)) + 2) for ki in k]):
            assert (idx_dense in idense) == (np.dot(idx_dense, k) < l), (
                f"Assertion failed with\n k = {k}, l = {l},\n idx = {idx_dense},\n idx*k = {np.dot(idx_dense, k)}, "
                + f"\n (idx in i) = {idx_dense in idense},\n np.dot(idx, k) < l = {np.dot(idx_dense, k) < l}"
            )
            idx_sparse = dense_index_to_sparse(idx_dense)
            assert (idx_sparse in isparse) == (np.dot(idx_dense, k) < l), (
                f"Assertion failed with\n k = {k}, l = {l},\n idx = {idx_dense},\n idx*k = {np.dot(idx_dense, k)}, "
                + f"\n (idx in i) = {idx_sparse in isparse},\n np.dot(idx, k) < l = {np.dot(idx_dense, k) < l}"
            )


def test_equality_of_sparse_and_dense_indexsets():
    print(
        "Testing that the sparse and dense multi-index set implementations contain the same multi-indices."
    )

    for i in range(10):
        k, l, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for nu in isparse:
            nu_dense = sparse_index_to_dense(nu, cutoff=len(k))
            assert nu_dense in idense, nu_dense

        for nu in idense:
            nu_sparse = dense_index_to_sparse(nu)
            assert nu_sparse in isparse, nu_sparse


def test_smolyak_coefficients():
    print(
        "Testing that sparse and dense computation of the Smolyak coefficients coincide."
    )

    for i in range(10):
        k, l, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for nu_1, nu_2 in zip(isparse, idense):
            c_1 = fast_smolyak_coefficient_zeta_sparse(
                lambda j: k[j], l, nu=nu_1, cutoff=len(k)
            )
            c_2 = smolyak_coefficient_zeta_dense(k, l, nu=nu_2)
            assert c_1 == c_2
