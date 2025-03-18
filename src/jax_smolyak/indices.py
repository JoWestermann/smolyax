from typing import Mapping

import cykhash
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


def indexset_sparse_w_cutoff(k, t, cutoff, i=0, idx=None):
    if idx is None:
        idx = {}
    if i >= cutoff:
        return [idx]
    r = []
    if (i + 1 < cutoff) and k(i + 1) < t:
        r += indexset_sparse_w_cutoff(k, t, cutoff, i + 1, idx)
    else:
        r += [idx]
    j = 1
    while j * k(i) < t:
        if i not in idx:  # Only allow `i: j` if it hasn't been assigned
            idx[i] = j  # Assign `i: j`
            r += indexset_sparse_w_cutoff(k, t - j * k(i), cutoff, i + 1, {**idx, i: j})
            del idx[i]  # Restore state after recursion
        j += 1
    return r


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
        if i not in idx:  # Only allow `i: j` if it hasn't been assigned
            idx[i] = j  # Assign `i: j`
            r += indexset_sparse(k, t - j * k(i), i + 1, {**idx, i: j}, cutoff=cutoff)
            del idx[i]  # Restore state after recursion
        j += 1
    return r


def count_without_constructing_idx_set(k, t, cutoff, i=0, idx=None):
    """Optimized recursive count of sparse index sets while preventing redundancy."""

    if idx is None:
        idx = cykhash.Int32toInt32Map()

    # Base case: If depth is exceeded, count as valid
    if i >= cutoff:
        return 1

    count_val = 0

    # Case 1: Skip k(i) and move to the next index
    if (i + 1 < cutoff) and k(i + 1) < t:
        count_val += count_without_constructing_idx_set(k, t, cutoff, i + 1, idx)
    else:
        count_val += 1

    # Case 2: Include multiples of k(i) and recurse
    j = 1
    while j * k(i) < t:
        if i not in idx:  # Only allow `i: j` if it hasn't been assigned
            idx[i] = j  # Assign `i: j`
            count_val += count_without_constructing_idx_set(
                k, t - j * k(i), cutoff, i + 1, idx
            )
            del idx[i]  # Restore state after recursion
        j += 1

    return count_val


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


def abs_e_sparse_sum(k, t, cutoff):
    """Computes sum((-1)^e) inline, correctly including all values of e."""
    stack = [(0, t, 0)]  # Stack holds (i, remaining t, e)
    result_sum = 0  # Accumulates sum((-1)^e)

    while stack:
        i, t_rem, e = stack.pop()

        # Ensure we count *every* valid `e`, even when `e=0`
        if i >= cutoff or t_rem <= 0:
            result_sum += 1 - 2 * (e & 1)
            continue

        k_i = k(i)  # Avoid redundant function calls

        # Case 1: Include k(i) in the sum if it's valid
        if k_i < t_rem:
            stack.append((i + 1, t_rem - k_i, e + 1))  # Increment `e`

        # Case 2: Skip k(i) and move to the next index
        stack.append((i + 1, t_rem, e))  # Keep `e` the same

    return result_sum


def cardinality_of_multiindex(k, t, nu, cutoff):
    """Handles `nu` initialization and calls the optimized sum computation."""
    t -= np.sum(nu[j] * k(j) for j in nu.keys())  # Apply `nu` adjustment
    return abs_e_sparse_sum(k, t, cutoff)  # Directly compute sum inline


def count_abs_e_sparse_fast_pow_w_cutoff(k, t, cutoff, i=0, e=None, *, nu=None) -> int:
    """Optimized version of abs_e_sparse applying `& 1` inline."""
    if e is None:
        e = 0
        t -= sum(nu[j] * k(j) for j in nu)  # Precompute modified `t`

    if i >= cutoff:
        return 1 - 2 * (e & 1)  # Directly compute (-1)^e

    count_val = 0  # Accumulator for sum

    # Case 1: Skip k(i) and move to the next index
    i_plus_1 = i + 1
    if (i_plus_1 < cutoff) and k(i_plus_1) < t:
        count_val += count_abs_e_sparse_fast_pow_w_cutoff(k, t, cutoff, i_plus_1, e)

    else:
        count_val += 1 - 2 * (e & 1)  # Inline computation of (-1)^e

    # Case 2: Include k(i) and recurse
    k_i = k(i)
    if k_i < t:
        count_val += count_abs_e_sparse_fast_pow_w_cutoff(
            k, t - k_i, cutoff, i_plus_1, e + 1
        )

    return count_val


def smolyak_coefficient_zeta_sparse(k, t, *, nu=None, cutoff=None) -> int:
    return np.sum([(-1) ** e for e in abs_e_sparse(k, t, nu=nu, cutoff=cutoff)])


def fast_smolyak_coefficient_zeta_sparse(k, t, *, nu=None, cutoff=None) -> int:
    return count_abs_e_sparse_fast_pow_w_cutoff(k, t, cutoff, nu=nu)


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
    if nested:
        return count_without_constructing_idx_set(kmap, t, cutoff)
    else:
        indices = indexset_sparse(kmap, t, cutoff=cutoff)
        total = np.sum(
            [
                np.prod([v + 1 for v in nu.values()])
                for nu in indices
                if not all((x == 0 for x in nu.values()))  # Skip only if all zeros
            ],
            dtype=np.int32,
        )

        # n = 0
        # for nu in indices:
        #     c = np.sum([(-1) ** e for e in abs_e_sparse(kmap, t, nu=nu, cutoff=cutoff)])
        #     if c != 0:
        #         n += np.prod([v + 1 for v in nu.values()])

        # cardinalities2 = [
        #     np.prod([v + 1 for v in nu.values()])
        #     for nu in indices
        #     if cardinality_of_multiindex(kmap, t, nu, cutoff) != 0
        # ]
        # assert cardinalit == np.sum(cardinalities2, dtype=np.int32), f"{cardinalit}, {np.sum(cardinalities2, dtype=np.int32)}"
        return total


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
