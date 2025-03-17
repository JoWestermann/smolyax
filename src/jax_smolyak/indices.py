from typing import Mapping

import numpy as np
import cykhash
# deprecated


def indexset_dense(k, l, idx=None):
    if idx is None:
        idx = ()
    if len(k) == 0:
        return [idx]
    r = []
    j = 0
    while j * k[0] < l:
        r += indexset_dense(k[1:], l - j * k[0], idx + (j,))
        j += 1
    return r


# deprecated
def unitball(nu, k, l, e=None):
    if e is None:
        e = []
    if len(k) == 0:
        return [e]
    r = unitball(nu[1:], k[1:], l - nu[0] * k[0], e + [0])
    if np.dot(nu, k) + k[0] < l:
        r += unitball(nu[1:], k[1:], l - (nu[0] + 1) * k[0], e + [1])
    return r


def smolyak_coefficient_zeta_dense(k, l, *, nu=None):
    return np.sum([(-1) ** (np.sum(e)) for e in unitball(nu, k, l)])


def indexset_sparse(k, l, i=0, idx=None, *, cutoff=None):
    if idx is None:
        idx = {}
    if cutoff is not None and i >= cutoff:
        return [idx]
    r = []
    if (cutoff is None or i + 1 < cutoff) and k(i + 1) < l:
        r += indexset_sparse(k, l, i + 1, idx, cutoff=cutoff)
    else:
        r += [idx]
    j = 1
    while j * k(i) < l:
        r += indexset_sparse(k, l - j * k(i), i + 1, {**idx, i: j}, cutoff=cutoff)
        j += 1
    return r


def fast_indexset_sparse_count_w_cutoff(k, l, cutoff, i=0, idx=None):
    """Optimized recursive count of sparse index sets while preventing redundancy."""

    if idx is None:
        idx = cykhash.Int32toInt32Map()

    # Base case: If depth is exceeded, count as valid
    if i >= cutoff:
        return 1

    count_val = 0

    # Case 1: Skip k(i) and move to the next index
    if (i + 1 < cutoff) and k(i + 1) < l:
        count_val += fast_indexset_sparse_count_w_cutoff(k, l, cutoff, i + 1, idx)
    else:
        count_val += 1

    # Case 2: Include multiples of k(i) and recurse
    j = 1
    while j * k(i) < l:
        if i not in idx:  # Only allow `i: j` if it hasn't been assigned
            idx[i] = j  # Assign `i: j`
            count_val += fast_indexset_sparse_count_w_cutoff(k, l - j * k(i), cutoff, i + 1, idx)
            del idx[i]  # Restore state after recursion
        j += 1

    return count_val


def abs_e_sparse(k, l, i=0, e=None, *, nu=None, cutoff=None):
    if e is None:
        assert i == 0 and nu is not None
        e = 0
        l -= np.sum([nu[j] * k(j) for j in nu.keys()])
    if cutoff is not None and i >= cutoff:
        return [e]
    r = []
    if (cutoff is None or i + 1 < cutoff) and k(i + 1) < l:
        r += abs_e_sparse(k, l, i + 1, e, cutoff=cutoff)
    else:
        r += [e]
    if k(i) < l:
        r += abs_e_sparse(k, l - k(i), i + 1, e + 1, cutoff=cutoff)
    return r


def count_abs_e_sparse_fast_pow_w_cutoff(k, l, cutoff, i=0, e=None, *, nu=None):
    """Optimized version of abs_e_sparse applying `& 1` inline."""
    if e is None:
        # assert i == 0 and nu is not None
        e = 0
        # print(nu)  # Debugging print (optional)
        l -= sum(nu[j] * k(j) for j in nu)  # Precompute modified `l`

    if i >= cutoff:
        return 1 - 2 * (e & 1)  # Directly compute (-1)^e

    count_val = 0  # Accumulator for sum

    # Case 1: Skip k(i) and move to the next index
    i_plus_1 = i + 1
    if (i_plus_1 < cutoff) and k(i_plus_1) < l:
        count_val += count_abs_e_sparse_fast_pow_w_cutoff(k, l, cutoff, i_plus_1, e)

    else:
        count_val += 1 - 2 * (e & 1)  # Inline computation of (-1)^e

    # Case 2: Include k(i) and recurse
    k_i = k(i)
    if k_i < l:
        count_val += count_abs_e_sparse_fast_pow_w_cutoff(k, l - k_i, cutoff, i_plus_1, e + 1)

    return count_val


def smolyak_coefficient_zeta_sparse(k, l, *, nu=None, cutoff=None) -> int:
    return np.sum([(-1) ** e for e in abs_e_sparse(k, l, nu=nu, cutoff=cutoff)])


def fast_smolyak_coefficient_zeta_sparse(k, l, *, nu=None, cutoff=None) -> int:
    return count_abs_e_sparse_fast_pow_w_cutoff(k, l, cutoff, nu=nu)


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


def n_points(kmap, l, cutoff, nested: bool = False) -> int:
    if nested:
        return fast_indexset_sparse_count_w_cutoff(kmap, l, cutoff)
    else:
        iset = indexset_sparse(kmap, l, cutoff=cutoff)
        n = 0
        for nu in iset:
            c = np.sum(1 - 2 * (np.array(abs_e_sparse(kmap, l, nu=nu, cutoff=cutoff)) & 1))
            print("computed c")
            if c != 0:
                print(nu.values(), "nu values")
                n += np.prod([v + 1 for v in nu.values()])
        return n


def find_suitable_l(k: Mapping, n: int = 50, nested: bool = False) -> int:
    assert n > 0

    def kmap(j):
        return k[j]

    cutoff = len(k)

    if n == 1:
        return 1

    # establish search interval
    l_interval = [1, 2]
    while n_points(kmap, l_interval[0], cutoff, nested) > n:
        l_interval[0] /= 1.2
    while n_points(kmap, l_interval[1], cutoff, nested) < n:
        l_interval[1] *= 1.2

    # bisect search interval
    def midpoint(interval):
        return interval[0] + (interval[1] - interval[0]) / 2

    l_cand = midpoint(l_interval)
    n_cand = n_points(kmap, l_cand, cutoff, nested)
    for _ in range(32):
        if n_cand > n:
            l_interval = [l_interval[0], l_cand]
        else:
            l_interval = [l_cand, l_interval[1]]
        l_cand = midpoint(l_interval)
        n_cand = n_points(kmap, l_cand, cutoff, nested)
        if n_cand == n:
            break
    return l_cand
