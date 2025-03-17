from typing import Mapping

import numpy as np


# deprecated
def indexset_dense(k, t, idx=None):
    if idx is None:
        idx = ()
    if len(k) == 0:
        return [idx]
    r = []
    j = 0
    while j * k[0] < t:
        r += indexset_dense(k[1:], t - j * k[0], idx + (j,))
        j += 1
    return r


# deprecated
def unitball(nu, k, t, e=None):
    if e is None:
        e = []
    if len(k) == 0:
        return [e]
    r = unitball(nu[1:], k[1:], t - nu[0] * k[0], e + [0])
    if np.dot(nu, k) + k[0] < t:
        r += unitball(nu[1:], k[1:], t - (nu[0] + 1) * k[0], e + [1])
    return r


def smolyak_coefficient_zeta_dense(k, t, *, nu=None):
    return np.sum([(-1) ** (np.sum(e)) for e in unitball(nu, k, t)])


def indexset_sparse(k, t, i=0, idx=None, *, cutoff=None):
    if idx is None:
        idx = {}
    if cutoff is not None and i >= cutoff:
        return [idx]
    r = []
    if (cutoff is None or i + 1 < cutoff) and k(i + 1) < t:
        r += indexset_sparse(k, t, i + 1, idx, cutoff=cutoff)
    else:
        r += [idx]
    j = 1
    while j * k(i) < t:
        r += indexset_sparse(k, t - j * k(i), i + 1, {**idx, i: j}, cutoff=cutoff)
        j += 1
    return r


def abs_e_sparse(k, t, i=0, e=None, *, nu=None, cutoff=None):
    if e is None:
        assert i == 0 and nu is not None
        e = 0
        t -= np.sum([nu[j] * k(j) for j in nu.keys()])
    if cutoff is not None and i >= cutoff:
        return [e]
    r = []
    if (cutoff is None or i + 1 < cutoff) and k(i + 1) < t:
        r += abs_e_sparse(k, t, i + 1, e, cutoff=cutoff)
    else:
        r += [e]
    if k(i) < t:
        r += abs_e_sparse(k, t - k(i), i + 1, e + 1, cutoff=cutoff)
    return r


def smolyak_coefficient_zeta_sparse(k, t, *, nu=None, cutoff=None):
    return np.sum([(-1) ** e for e in abs_e_sparse(k, t, nu=nu, cutoff=cutoff)])


def sparse_index_to_dense(nu, cutoff=None) -> tuple:
    if cutoff is None:
        cutoff = max(nu.keys())
    dense_nu = [0] * cutoff
    for k, v in nu.items():
        dense_nu[k] = v
    return tuple(dense_nu)


def dense_index_to_sparse(dense_nu):
    sparse_nu = {}
    for k, v in enumerate(dense_nu):
        if v > 0:
            sparse_nu[k] = v
    return sparse_nu


def cardinality(kmap, t, cutoff, nested: bool = False) -> int:
    iset = indexset_sparse(kmap, t, cutoff=cutoff)

    if nested:
        return len(iset)

    n = 0
    for nu in iset:
        c = np.sum([(-1) ** e for e in abs_e_sparse(kmap, t, nu=nu, cutoff=cutoff)])
        if c != 0:
            n += np.prod([v + 1 for v in nu.values()])
    return n


def find_suitable_t(k: Mapping, m: int = 50, nested: bool = False) -> int:
    """
    k : weight vector of the anisotropy of the multi-index set
    m : target cardinality of the multi-index set
    returns t : threshold parameter to construct a k-weighted multi-index set of size (roughly) m
    """
    assert m > 0

    def kmap(j):
        return k[j]

    cutoff = len(k)

    if m == 1:
        return 1

    # establish search interval
    l_interval = [1, 2]
    while cardinality(kmap, l_interval[0], cutoff, nested) > m:
        l_interval[0] /= 1.2
    while cardinality(kmap, l_interval[1], cutoff, nested) < m:
        l_interval[1] *= 1.2

    # bisect search interval
    def midpoint(interval):
        return interval[0] + (interval[1] - interval[0]) / 2

    t_cand = midpoint(l_interval)
    m_cand = cardinality(kmap, t_cand, cutoff, nested)
    for _ in range(32):
        if m_cand > m:
            l_interval = [l_interval[0], t_cand]
        else:
            l_interval = [t_cand, l_interval[1]]
        t_cand = midpoint(l_interval)
        m_cand = cardinality(kmap, t_cand, cutoff, nested)
        if m_cand == m:
            break
    return t_cand
