from typing import Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike


def indexset(k, t: float):
    stack = [(0, t, ())]
    result = []

    while stack:
        dim_i, remaining_t, nu_head = stack.pop()

        # Check if the index nu_head is final

        if dim_i >= len(k) or k[dim_i] >= remaining_t:
            result.append(nu_head)
            continue

        # Otherwise add all possible extensions of nu_head onto the stack

        stack.append((dim_i + 1, remaining_t, nu_head))

        j = 1
        while j * k[dim_i] < remaining_t:
            nu_extended = nu_head + ((dim_i, j),)
            new_t = remaining_t - j * k[dim_i]
            stack.append((dim_i + 1, new_t, nu_extended))
            j += 1

    return result


@njit(cache=True)
def indexset_size(k, t):
    stack = [(0, 0.0)]
    count = 0

    while stack:
        dim_i, used_t = stack.pop()

        if dim_i >= len(k):
            count += 1
            continue

        remaining_t = t - used_t

        if dim_i + 1 < len(k) and k[dim_i + 1] < remaining_t:
            stack.append((dim_i + 1, used_t))
        else:
            count += 1

        j = 1
        while used_t + j * k[dim_i] < t:
            new_used_t = used_t + j * k[dim_i]
            stack.append((dim_i + 1, new_used_t))
            j += 1

    return count


def abs_e(k, t, i=0, e=None, *, nu: dict[int, int] = None):
    if e is None:
        assert i == 0 and nu is not None
        e = 0
        t -= np.sum([v * k[j] for j, v in nu])
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


@njit(cache=True)
def _abs_e_subtree_stack(k, d, rem_t, parity):
    """
    Suffix sum of (-1)^e exactly matching abs_e_list:
    always recurse from i=0 over all dimensions.
    """
    total = 0
    stack = [(0, rem_t, parity)]
    while stack:
        i, rt, p = stack.pop()
        if i >= d:
            total += 1 - (p << 1)
            continue
        # skip‐case
        if i + 1 < d and k[i + 1] < rt:
            stack.append((i + 1, rt, p))
        else:
            total += 1 - (p << 1)
        # include‐case (one copy)
        cost = k[i]
        if cost < rt:
            stack.append((i + 1, rt - cost, p ^ 1))
    return total


@njit(cache=True)
def non_nested_cardinality(k, t):
    """
     For each nu in indexset(k,t):
       if sum((-1)**e for e in abs_e_list(k,t,nu)) != 0
         add prod(v+1 for (_,v) in nu)
    —all in one Nopython pass.
    """
    d = k.shape[0]
    total = 0
    # main stack holds (dim_i, rem_budget, parity, prod_n)
    stack = [(0, t, 0, 1)]
    while stack:
        dim_i, rem_t, parity, prod_n = stack.pop()

        # terminal skip‐branch?
        if dim_i >= d or not (dim_i + 1 < d and k[dim_i + 1] < rem_t):
            s = _abs_e_subtree_stack(k, d, rem_t, parity)
            if s != 0:
                total += prod_n

        # now expand exactly like your original indexset
        if dim_i < d:
            # skip‐branch
            if dim_i + 1 < d and k[dim_i + 1] < rem_t:
                stack.append((dim_i + 1, rem_t, parity, prod_n))
            # include‐branches for all j≥1
            cost = k[dim_i]
            j = 1
            while cost * j < rem_t:
                new_parity = parity ^ (j & 1)
                new_prod_n = prod_n * (j + 1)
                new_rem_t = rem_t - cost * j
                stack.append((dim_i + 1, new_rem_t, new_parity, new_prod_n))
                j += 1

    return total


def smolyak_coefficient_zeta(k, t: float, *, nu: dict[int, int] = None):
    return np.sum([(-1) ** e for e in abs_e(k, t, nu=nu)])


def sparse_index_to_dense(nu: dict[int, int], dim: int) -> tuple:
    dense_nu = [0] * dim
    for k, v in nu:
        dense_nu[k] = v
    return tuple(dense_nu)


def dense_index_to_sparse(dense_nu: ArrayLike) -> Tuple[Tuple[int, int], ...]:
    sparse_nu = []
    for k, v in enumerate(dense_nu):
        if v > 0:
            sparse_nu.append((k, v))
    return tuple(sparse_nu)


def cardinality(k, t: float, nested: bool = False) -> int:
    if nested:
        return indexset_size(k, t)
    return non_nested_cardinality(k, t)


def find_approximate_threshold(k: ArrayLike, m: int, nested: bool, max_iter: int = 32, accuracy: float = 0.001) -> int:
    """
    Find the approximate threshold parameter to construct a k-weighted multi-index set such that the set of
    corresponding interpolation nodes has a cardinality of approximately `m`.

    Parameters
    ----------
    k : ArrayLike
        Weight vector of the anisotropy of the multi-index set.
    m : int
        Target cardinality of the set of interpolation nodes.
    nested : bool
        Flag to indicate whether nested or non-nested interpolation nodes are used.
    max_iter : int, optional
        Maximal number of bisection iterations. Default is 32.
    accuracy : float, optional
        Relative tolerance within which the cardinality of the set of interpolation nodes may deviate from `m`.
        Note that the accuracy may not be reached if the maximum number of iterations `max_iter` is exhausted.
        Default is 0.001.

    Returns
    -------
    int
        Threshold parameter `t` to construct a k-weighted multi-index set of size approximately `m`.

    Notes
    -----
    * The function uses a bisection method to find the threshold parameter `t` such that the cardinality
      of the set of interpolation nodes is approximately equal to `m`.
    * When `nested` is True, the cardinality of the index set is equal to the cardinality of the set of
      interpolation nodes.
    """
    assert m > 0

    if m == 1:
        return 1

    # establish search interval
    l_interval = [1.0, 2.0]
    while cardinality(k, l_interval[0], nested) > m:
        l_interval = [l_interval[0] / 1.2, l_interval[0]]
    while cardinality(k, l_interval[1], nested) < m:
        l_interval = [l_interval[1], l_interval[1] * 1.2]

    # bisect search interval
    def midpoint(interval):
        return interval[0] + (interval[1] - interval[0]) / 2.0

    t_cand = midpoint(l_interval)
    m_cand = cardinality(k, t_cand, nested)
    for _ in range(max_iter):
        if m_cand > m:
            l_interval = [l_interval[0], t_cand]
        else:
            l_interval = [t_cand, l_interval[1]]
        t_cand = midpoint(l_interval)
        m_cand = cardinality(k, t_cand, nested)

        if np.abs(m_cand - m) / m < accuracy:
            break

    return t_cand
