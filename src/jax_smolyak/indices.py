from collections import defaultdict
from typing import Sequence

import numpy as np
from numba import njit


def sparse_index_to_dense(nu: tuple[tuple[int, int], ...], dim: int) -> tuple:
    dense_nu = [0] * dim
    for k, v in nu:
        dense_nu[k] = v
    return tuple(dense_nu)


def dense_index_to_sparse(dense_nu: tuple[int]) -> tuple[tuple[int, int], ...]:
    sparse_nu = []
    for k, v in enumerate(dense_nu):
        if v > 0:
            sparse_nu.append((k, v))
    return tuple(sparse_nu)


def indexset(k: Sequence[float], t: float):
    r"""
    Generate the `k`-weighted anisotropic multi-index set $\Lambda_{\boldsymbol{k}, t}$ with a given threshold `t`,
    which is defined as
    $$
    \Lambda_{\boldsymbol{k}, t} := \\{\boldsymbol{\nu} \in \mathbb{N}_0^{d_1} \ : \ \sum_{j=1}^{d_1} k_j \nu_j < t\\}.
    $$

    Parameters
    ----------
    k : Sequence[float]
        Weight vector of the anisotropy of the multi-index set.
    t : float
        Threshold parameter to control the size of the multi-index set.

    Returns
    -------
    list of tuples
        A list of multi-indices that satisfy the k-weighted condition with threshold `t`. The multi-indices are given in
        a tuple-based sparse format given as $((j, \nu_j))_{j \in \{0, \dots, d\} : \nu_j > 0}$.

    Notes
    -----
    * To find a suitable threshold parameter `t` that allows to construct a `k`-weighted multi-index with a specified
        cardinality use [`find_approximate_threshold()`](#find_approximate_threshold).
    """
    d = len(k)
    stack = [(0, t, ())]  # dimension, threshold, multi-index head (entries in the first dimensions)
    result = []

    while stack:
        i, remaining_t, nu_head = stack.pop()

        # Check if the stack entry is final
        if i >= d or k[i] >= remaining_t:
            result.append(nu_head)
            continue

        # Add nu_head with nu_i = 0 on to the stack
        stack.append((i + 1, remaining_t, nu_head))

        # Add all admissible nu_head with nu_i = j on to the stack
        j = 1
        k_i = k[i]
        while j * k_i < remaining_t:
            nu_extended = nu_head + ((i, j),)
            new_t = remaining_t - j * k_i
            stack.append((i + 1, new_t, nu_extended))
            j += 1

    return result


@njit(cache=True)
def indexset_size(k, t):
    d = len(k)
    stack = [(0, t)]  # dimension, threshold
    count = 0

    while stack:
        i, remaining_t = stack.pop()

        # Check if the stack entry is final
        if i >= d or k[i] >= remaining_t:
            count += 1
            continue

        # Add case nu_i = 0 on to the stack
        stack.append((i + 1, remaining_t))

        # Add all admissible cases with nu_i = j on to the stack
        j = 1
        k_i = k[i]
        while j * k_i < remaining_t:
            new_t = remaining_t - j * k_i
            stack.append((i + 1, new_t))
            j += 1

    return count


# deprecated
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


# deprecated
def smolyak_coefficient_zeta(k, t: float, *, nu: dict[int, int] = None):
    return np.sum([(-1) ** e for e in abs_e(k, t, nu=nu)])


@njit(cache=True)
def _abs_e_subtree_stack(k, d, rem_t, parity):
    """
    Suffix sum of (-1)^e exactly matching abs_e_list:
    always recurse from i=0 over all dimensions.
    """
    total = 0
    stack = [(0, rem_t, parity)]  # dimension, threshold, parity

    while stack:
        i, rt, p = stack.pop()

        # Check if the stack entry is final
        if i >= d or k[i + 1] >= rt:
            total += 1 - (p << 1)
            continue

        # Add case e_i = 0 on to the stack
        stack.append((i + 1, rt, p))

        # Add case e_i = 1 on to the stack if admissible
        k_i = k[i]
        if k_i < rt:
            stack.append((i + 1, rt - k_i, p ^ 1))
    return total


@njit(cache=True)
def non_nested_cardinality(k, t):
    """
    For each nu in indexset(k,t):
    if sum((-1)**e for e in abs_e_list(k,t,nu)) != 0
    add prod(v+1 for (_,v) in nu)
    —all in one Nopython pass.
    """
    d = len(k)
    stack = [(0, t, 0, 1)]  # dimension, threshold, parity, prod_n
    total = 0

    while stack:
        i, rem_t, parity, prod_n = stack.pop()

        # Check if the stack entry is final
        if i >= len(k) or k[i] >= rem_t:
            zeta = _abs_e_subtree_stack(k, d, rem_t, parity)
            if zeta != 0:
                total += prod_n
            continue

        # Add case nu_i = 0 on to the stack
        stack.append((i + 1, rem_t, parity, prod_n))

        # Add all admissible cases with nu_i = j on to the stack
        j = 1
        k_i = k[i]
        while j * k_i < rem_t:
            new_parity = parity ^ (j & 1)
            new_prod_n = prod_n * (j + 1)
            new_rem_t = rem_t - j * k_i
            stack.append((i + 1, new_rem_t, new_parity, new_prod_n))
            j += 1

    return total


@njit(cache=True)
def _subtree_zeta(k, d, rem_t):
    stack = [(0, rem_t, 0)]  # dimension, threshold, zeta
    total = 0

    while stack:
        i, rt, e = stack.pop()

        # Check if the stack entry is final
        if i >= d or k[i] >= rt:
            total += 1 - ((e & 1) << 1)
            continue

        # Add case e_i = 0 on to the stack
        stack.append((i + 1, rt, e))

        # Add case e_i = 1 on to the stack if admissible
        if k[i] < rt:
            stack.append((i + 1, rt - k[i], e ^ 1))
    return total


def non_zero_indices_and_zetas(k, t):
    """
    Constructs the non-zero coefficient indices and their coefficeints in one DFS:
     – similar to indexset(k,t)
     – at each 'terminal' nu, calls subtree_sum(rem_t)
     – if zeta != 0, groups nu by len(nu)
    """
    d = len(k)
    n2nus, n2zetas = defaultdict(list), defaultdict(list)

    stack = [(0, t, ())]
    while stack:
        i, rem_t, nu = stack.pop()

        # Check if the stack entry is final
        if i >= d or k[i] >= rem_t:
            zeta = _subtree_zeta(k, d, rem_t)
            if zeta != 0:
                n = len(nu)
                n2nus[n].append(nu)
                n2zetas[n].append(zeta)
            continue

        # Add case nu_i = 0 on to the stack
        stack.append((i + 1, rem_t, nu))

        # Add all admissible nu_head with nu_i = j on to the stack
        j = 1
        k_i = k[i]
        while j * k_i < rem_t:
            stack.append((i + 1, rem_t - j * k_i, nu + ((i, j),)))
            j += 1
    return n2nus, n2zetas


def cardinality(k, t: float, nested: bool = False) -> int:
    if nested:
        return indexset_size(k, t)
    return non_nested_cardinality(k, t)


def find_approximate_threshold(
    k: Sequence[float], m: int, nested: bool, max_iter: int = 32, accuracy: float = 0.001
) -> float:
    """
    Find the approximate threshold parameter to construct a k-weighted multi-index set such that the set of
    corresponding interpolation nodes has a cardinality of approximately `m`.

    Parameters
    ----------
    k : Sequence[float]
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
    float
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
