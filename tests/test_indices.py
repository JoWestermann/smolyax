import itertools as it

import numpy as np

from jax_smolyak import indices


def get_random_indexsets(nested: bool = False):
    d = np.random.randint(low=1, high=6)

    a = 1.1 + 2.9 * np.random.random()  # a \in [1.1,4.)
    b = 0.1 + 1.9 * np.random.random()  # b \in [0.1,3)
    k = np.log([a + b * i for i in range(d)]) / np.log(a)

    n_t = np.random.randint(low=1, high=100)
    t = indices.find_suitable_t(k, n_t, nested=nested)

    isparse = indices.indexset_sparse(k, t, cutoff=d)
    idense = indices.indexset_dense(k, t)
    print(
        f"\tConstructed {d}-dimensional multi-index sets with a={a}, b={b} and target n={n_t}. "
        + f"Sets are of cardinality {len(isparse)}."
    )
    return k, t, isparse, idense


def test_validity_of_indexsets():
    print("Testing that index sets contain the correct multi-indices and none extra.")

    for i in range(10):
        k, t, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for idx_dense in it.product(*[range(int(np.floor(ki)) + 2) for ki in k]):
            assert (idx_dense in idense) == (np.dot(idx_dense, k) < t), (
                f"Assertion failed with\n k = {k}, t = {t},\n idx = {idx_dense},\n idx*k = {np.dot(idx_dense, k)}, "
                + f"\n (idx in i) = {idx_dense in idense},\n np.dot(idx, k) < t = {np.dot(idx_dense, k) < t}"
            )
            idx_sparse = indices.dense_index_to_sparse(idx_dense)
            assert (idx_sparse in isparse) == (np.dot(idx_dense, k) < t), (
                f"Assertion failed with\n k = {k}, t = {t},\n idx = {idx_dense},\n idx*k = {np.dot(idx_dense, k)}, "
                + f"\n (idx in i) = {idx_sparse in isparse},\n np.dot(idx, k) < t = {np.dot(idx_dense, k) < t}"
            )


def test_equality_of_sparse_and_dense_indexsets():
    print("Testing that the sparse and dense multi-index set implementations contain the same multi-indices.")

    for i in range(10):
        k, _, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for nu in isparse:
            nu_dense = indices.sparse_index_to_dense(nu, cutoff=len(k))
            assert nu_dense in idense, nu_dense

        for nu in idense:
            nu_sparse = indices.dense_index_to_sparse(nu)
            assert nu_sparse in isparse, nu_sparse


def test_smolyak_coefficients():
    print("Testing that sparse and dense computation of the Smolyak coefficients coincide.")

    for i in range(10):
        k, t, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for nu_1, nu_2 in zip(isparse, idense):
            c_1 = indices.smolyak_coefficient_zeta_sparse(k, t, nu=nu_1, cutoff=len(k))
            c_2 = indices.smolyak_coefficient_zeta_dense(k, t, nu=nu_2)
            assert c_1 == c_2
