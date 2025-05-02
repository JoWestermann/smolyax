import itertools as it

import indices_dense_reference_impl as indices_dense
import numpy as np

from jax_smolyak import indices as indices_sparse


def get_random_indexsets(nested: bool = False):
    d = np.random.randint(low=1, high=6)

    a = 1.1 + 2.9 * np.random.random()  # a \in [1.1,4.)
    b = 0.1 + 1.9 * np.random.random()  # b \in [0.1,3)
    k = np.log([a + b * i for i in range(d)]) / np.log(a)

    n_t = np.random.randint(low=1, high=100)
    t = indices_sparse.find_approximate_threshold(k, n_t, nested=nested)

    isparse = indices_sparse.indexset(k, t)
    idense = indices_dense.indexset(k, t)
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
            idx_sparse = indices_sparse.dense_index_to_sparse(idx_dense)
            assert (idx_sparse in isparse) == (np.dot(idx_dense, k) < t), (
                f"Assertion failed with\n k = {k}, t = {t},\n idx = {idx_dense},\n idx*k = {np.dot(idx_dense, k)}, "
                + f"\n (idx in i) = {idx_sparse in isparse},\n np.dot(idx, k) < t = {np.dot(idx_dense, k) < t}"
            )


def test_equality_of_sparse_and_dense_indexsets():
    print("Testing that the sparse and dense multi-index set implementations contain the same multi-indices.")

    for i in range(10):
        k, _, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for nu in isparse:
            nu_dense = indices_sparse.sparse_index_to_dense(nu, dim=len(k))
            assert nu_dense in idense, nu_dense

        for nu in idense:
            nu_sparse = indices_sparse.dense_index_to_sparse(nu)
            assert nu_sparse in isparse, nu_sparse


def test_smolyak_coefficients():
    print("Testing that sparse and dense computation of the Smolyak coefficients coincide.")

    for i in range(10):
        k, t, isparse, idense = get_random_indexsets(nested=(i % 2) == 0)

        for nu_sparse in isparse:
            nu_dense = indices_sparse.sparse_index_to_dense(nu_sparse, dim=len(k))
            assert nu_dense in idense
            c_sparse = indices_sparse.smolyak_coefficient_zeta(k, t, nu=nu_sparse)
            c_dense = indices_dense.smolyak_coefficient_zeta(k, t, nu=nu_dense)
            assert c_sparse == c_dense, (
                f"Assertion failed for nu_sparse={nu_sparse}, c_sparse={c_sparse}"
                + " and nu_dense={nu_dense}, c_dense={c_dense}"
            )
