import numpy as np
from numpy.typing import ArrayLike


def indexset(k, t: float, i: int = 0, nu: dict[int, int] = None):
    if nu is None:
        nu = {}
    if i >= len(k):
        return [nu]
    r = []
    if i + 1 < len(k) and k[i + 1] < t:
        r += indexset(k, t, i + 1, nu)
    else:
        r += [nu]
    j = 1
    while j * k[i] < t:
        r += indexset(k, t - j * k[i], i + 1, {**nu, i: j})
        j += 1
    return r


def abs_e(k, t, i=0, e=None, *, nu: dict[int, int] = None):
    if e is None:
        assert i == 0 and nu is not None
        e = 0
        t -= np.sum([nu[j] * k[j] for j in nu.keys()])
    if i >= len(k):
        return [e]
    r = []
    if i + 1 < len(k) and k[i + 1] < t:
        r += abs_e(k, t, i + 1, e)
    else:
        r += [e]
    if k[i] < t:
        r += abs_e(k, t - k[i], i + 1, e + 1)
    return r


def smolyak_coefficient_zeta(k, t: float, *, nu: dict[int, int] = None):
    return np.sum([(-1) ** e for e in abs_e(k, t, nu=nu)])


def sparse_index_to_tuple(nu: dict[int, int], check: bool = False) -> tuple:
    if check:
        assert list(nu.keys()) == sorted(nu.keys())
    return tuple(nu.items())


def sparse_index_to_dense(nu: dict[int, int], dim: int) -> tuple:
    dense_nu = [0] * dim
    for k, v in nu.items():
        dense_nu[k] = v
    return tuple(dense_nu)


def dense_index_to_sparse(dense_nu: ArrayLike) -> dict[int, int]:
    sparse_nu = {}
    for k, v in enumerate(dense_nu):
        if v > 0:
            sparse_nu[k] = v
    return sparse_nu


def cardinality(k, t: float, nested: bool = False) -> int:
    iset = indexset(k, t)

    if nested:
        return len(iset)

    n = 0
    for nu in iset:
        c = np.sum([(-1) ** e for e in abs_e(k, t, nu=nu)])
        if c != 0:
            n += np.prod([v + 1 for v in nu.values()])
    return n


def find_suitable_t(k: ArrayLike, m: int = 50, nested: bool = False) -> int:
    """
    k : weight vector of the anisotropy of the multi-index set
    m : target cardinality of the multi-index set
    returns t : threshold parameter to construct a k-weighted multi-index set of size (roughly) m
    """
    assert m > 0

    if m == 1:
        return 1

    # establish search interval
    l_interval = [1, 2]
    while cardinality(k, l_interval[0], nested) > m:
        l_interval[0] /= 1.2
    while cardinality(k, l_interval[1], nested) < m:
        l_interval[1] *= 1.2

    # bisect search interval
    def midpoint(interval):
        return interval[0] + (interval[1] - interval[0]) / 2

    t_cand = midpoint(l_interval)
    m_cand = cardinality(k, t_cand, nested)
    for _ in range(32):
        if m_cand > m:
            l_interval = [l_interval[0], t_cand]
        else:
            l_interval = [t_cand, l_interval[1]]
        t_cand = midpoint(l_interval)
        m_cand = cardinality(k, t_cand, nested)
        if m_cand == m:
            break
    return t_cand
