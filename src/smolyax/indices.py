r"""
This module contains functionality to compute $\boldsymbol{k}$-weighted anisotropic multi-index sets
$\Lambda_{\boldsymbol{k}, t}$ and related data structures and quantities.

For a given weight vector $\boldsymbol{k} \in \mathbb{R}^d$ the anisotropic multi-index set
$\Lambda_{\boldsymbol{k}, t} \subset \mathbb{N}_0^d$ is defined as
$$
\Lambda_{\boldsymbol{k}, t} := \\{\boldsymbol{\nu} \in \mathbb{N}_0^{d} \ : \ \sum_{j=1}^{d} k_j \nu_j < t\\}.
$$
where $t > 0$ is a scalar threshold parameter that allows to control the size of the multi-index set.
"""

from collections import defaultdict
from typing import Sequence

import numpy as np
from numba import njit


def indexset(
    k: Sequence[float],
    t: float,
    quadrature: bool = False,
) -> list[tuple[tuple[int, int]]]:
    r"""
    Generate the $\boldsymbol{k}$-weighted anisotropic multi-index set
    $\Lambda_{\boldsymbol{k}, t} \subset \mathbb{N}_0^d$ with threshold $t$.

    Parameters
    ----------
    k : Sequence[float]
        Weight vector of the anisotropy of the multi-index set. The dimension $d$ is inferred as `len(k)`.
    t : float
        Threshold parameter to control the cardinality of the multi-index set.
    quadrature : bool, optional
        If True, the multi-index set is generated for quadrature purposes, which may affect the construction process.

    Returns
    -------
    list of tuples
        A list of multi-indices that satisfy the k-weighted condition with threshold `t`. The multi-indices are given in
        a tuple-based sparse format given as $((j, \nu_j))_{j \in \\{0, \dots, d\\} : \nu_j > 0\}$.

    Notes
    -----
    * To compute the cardinality of the set efficiently without constructing it use
        [`indexset_cardinality()`](#indexset_cardinality).
    * To find a suitable threshold parameter `t` that allows to construct a `k`-weighted multi-index with a specified
        cardinality use [`find_approximate_threshold()`](#find_approximate_threshold) with `nested = True`.
    """

    def cost_multiplier(j: int) -> int:
        if quadrature and j == 1:
            return 2
        return j

    d = len(k)
    stack = [(0, t, ())]  # dimension, threshold, multi-index head (entries in the first dimensions)
    result = []

    while stack:
        i, remaining_t, nu_head = stack.pop()

        # For standard index sets, the cheapest positive value is nu_i = 1,
        # with cost k_i.
        #
        # For quadrature index sets, the cheapest positive values are
        # nu_i = 1 and nu_i = 2, both with cost 2 * k_i.
        min_positive_multiplier = 2 if quadrature else 1

        # Check if the stack entry is final
        if i >= d or min_positive_multiplier * k[i] >= remaining_t:
            result.append(nu_head)
            continue

        # Add nu_head with nu_i = 0 on to the stack
        stack.append((i + 1, remaining_t, nu_head))

        # Add all admissible nu_head with nu_i = j on to the stack
        j = 1
        k_i = k[i]
        while cost_multiplier(j) * k_i < remaining_t:
            nu_extended = nu_head + ((i, j),)
            new_t = remaining_t - cost_multiplier(j) * k_i
            stack.append((i + 1, new_t, nu_extended))
            j += 1

    return result


@njit(cache=True)
def indexset_cardinality(k: Sequence[float], t: float, quadrature: bool = False) -> int:
    r"""
    Compute the cardinality of the $\boldsymbol{k}$-weighted anisotropic multi-index set
    $\Lambda_{\boldsymbol{k}, t} \subset \mathbb{N}_0^d$ with threshold $t$.

    Parameters
    ----------
    k : Sequence[float]
        Weight vector of the anisotropy of the multi-index set. The dimension $d$ is inferred as `len(k)`.
    t : float
        Threshold parameter to control the cardinality of the multi-index set.
    quadrature : bool, optional
        If True, the cardinality is computed for quadrature purposes.

    Returns
    -------
    int
        Cardinality of the multi-index set $\Lambda_{\boldsymbol{k}, t}.$

    Notes
    -----
    * The result of this method is equivalent to `len(indexset(k, t))`, but it is computed significantly more
        efficiently since the index set is not explicitly constructed and numba compilation is used."
    """

    def cost_multiplier(j: int) -> int:
        return 2 if quadrature and j == 1 else j

    min_positive_multiplier = 2 if quadrature else 1

    stack = [(0, 0.0)]  # dimension, threshold
    count = 0

    while stack:
        dim_i, used_t = stack.pop()

        if dim_i >= len(k):
            count += 1
            continue

        remaining_t = t - used_t

        if dim_i + 1 < len(k) and min_positive_multiplier * k[dim_i + 1] < remaining_t:
            stack.append((dim_i + 1, used_t))
        else:
            count += 1

        j = 1
        while used_t + cost_multiplier(j) * k[dim_i] < t:
            new_used_t = used_t + cost_multiplier(j) * k[dim_i]
            stack.append((dim_i + 1, new_used_t))
            j += 1

    return count


@njit(cache=True)
def smolyak_coefficient(k: Sequence[float], d: int, rem_t: float, parity: int) -> int:
    r"""
    Computes the smolyak coefficient $\zeta_{\Lambda_{\boldsymbol{k},t}, \boldsymbol{\nu}} := \sum
    \limits_{\boldsymbol{e} \in \\{0,1\\}^d : \boldsymbol{\nu}+\boldsymbol{e} \in \Lambda_{\boldsymbol{k},t}}
    (-1)^{|\boldsymbol{e}|}$.

    However, instead of taking $\boldsymbol{\nu}$ and checking for every $\boldsymbol{e} \in \\{0,1\\}^d$ whether
    $\sum_{j=1}^d (\nu_j + e_j) k_j < t$, the function takes the *remaining* threshold value
    $\tilde{t}_{t, \boldsymbol{k}, \boldsymbol{\nu}} := t - \sum_{j=1}^d \nu_j k_j$.
    and then checks for all $\boldsymbol{e}$ whether $\sum_{j=1}^d e_j k_j < t_{\boldsymbol{k}, \boldsymbol{\nu}}.$

    For efficiency, the indices $\boldsymbol{e}$ are not constructed explicitly. Instead, the parity of the exponent
    $|\boldsymbol{e}|$ is tracked by bit-flips while iterating over the dimensions $j \in \\{0, \dots, d\\}.q$

    Parameters
    ----------
    k : Sequence[float]
        Weight vector of the anisotropy of the multi-index set.
    d : int
        Number of dimensions (equal to `len(k)`)
    rem_t : float
        Remaining threshold value $\tilde{t}_{t, \boldsymbol{k}, \boldsymbol{\nu}} := t - \sum_{j=1}^d \nu_j k_j$
    parity : int
        Initial parity (0 or 1).

    Returns
    -------
    int
        The Smolyak coefficient $\zeta_{\Lambda_{\boldsymbol{k},t}, \boldsymbol{\nu}}.$
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
def smolyak_coefficient_quadrature(
    k: Sequence[float],
    d: int,
    rem_t: float,
    parity: int,
    nu_indices: np.ndarray,
    nu_values: np.ndarray,
) -> int:
    r"""
    Computes the Smolyak coefficient for the quadrature index set.

    The quadrature cost uses

        hat(nu_i) = 2 if nu_i == 1 else nu_i.

    Hence the incremental cost of adding e_i = 1 is

        nu_i = 0  -> 2 * k_i
        nu_i = 1  -> 0
        nu_i >= 2 -> k_i.
    """
    total = 0

    # entries are: dimension i, remaining threshold rt, parity p, position in sparse nu arrays
    stack = [(0, rem_t, parity, 0)]

    while stack:
        i, rt, p, pos = stack.pop()

        if i >= d:
            total += 1 - (p << 1)
            continue

        if pos < len(nu_indices) and nu_indices[pos] == i:
            nu_i = nu_values[pos]
            next_pos = pos + 1
        else:
            nu_i = 0
            next_pos = pos

        # skip-case: e_i = 0
        stack.append((i + 1, rt, p, next_pos))

        # include-case: e_i = 1
        if nu_i == 0:
            cost = 2.0 * k[i]
        elif nu_i == 1:
            cost = 0.0
        else:
            cost = k[i]

        if cost < rt:
            stack.append((i + 1, rt - cost, p ^ 1, next_pos))

    return total


def __nodeset_cardinality_nested(k: Sequence[float], t: float, quadrature: bool = False) -> int:
    return indexset_cardinality(k, t, quadrature=quadrature)


@njit(cache=True)
def __nodeset_cardinality_non_nested(k: Sequence[float], t: float) -> int:
    """
    For each nu in indexset(k,t):
    if sum((-1)**e for e in abs_e_list(k,t,nu)) != 0
    add prod(v+1 for (_,v) in nu)
    —all in one Nopython pass.
    """
    d = len(k)
    stack = [(0, t, 0, 1)]  # dimension, rem_budget, parity, prod_n
    total = 0

    while stack:
        dim_i, rem_t, parity, prod_n = stack.pop()

        # terminal skip‐branch?
        if dim_i >= d or not (dim_i + 1 < d and k[dim_i + 1] < rem_t):
            zeta = smolyak_coefficient(k, d, rem_t, parity)
            if zeta != 0:
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


@njit(cache=True)
def __nodeset_cardinality_non_nested_quadrature(
    k: Sequence[float],
    t: float,
) -> int:
    """
    Compute the cardinality of the non-nested node set associated with the
    quadrature multi-index set.

    For each nu in the quadrature index set:
        if zeta_nu != 0, add prod(v + 1 for (_, v) in nu).

    This mirrors __nodeset_cardinality_non_nested, but carries the sparse
    multi-index representation needed by smolyak_coefficient_quadrature.
    """
    d = len(k)

    def cost_multiplier(j: int) -> int:
        return 2 if j == 1 else j

    min_positive_multiplier = 2

    total = 0

    # Stack entries:
    #   dim_i      : current dimension
    #   rem_t      : remaining threshold
    #   parity     : current parity contribution
    #   prod_n     : product of (nu_i + 1) over active dimensions
    #   nu_indices : active dimensions of current sparse nu
    #   nu_values  : corresponding positive values of current sparse nu
    stack = [
        (
            0,
            t,
            0,
            1,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    ]

    while stack:
        dim_i, rem_t, parity, prod_n, nu_indices, nu_values = stack.pop()

        has_later_positive_extension = dim_i + 1 < d and min_positive_multiplier * k[dim_i + 1] < rem_t

        # terminal skip-branch?
        if dim_i >= d or not has_later_positive_extension:
            zeta = smolyak_coefficient_quadrature(
                k,
                d,
                rem_t,
                parity,
                nu_indices,
                nu_values,
            )

            if zeta != 0:
                total += prod_n

        # expand exactly like the quadrature indexset
        if dim_i < d:
            # skip-branch: nu_dim_i = 0
            if has_later_positive_extension:
                stack.append(
                    (
                        dim_i + 1,
                        rem_t,
                        parity,
                        prod_n,
                        nu_indices,
                        nu_values,
                    )
                )

            # include-branches: nu_dim_i = j >= 1
            j = 1
            while cost_multiplier(j) * k[dim_i] < rem_t:
                cost = cost_multiplier(j) * k[dim_i]

                new_parity = parity ^ (j & 1)
                new_prod_n = prod_n * (j + 1)
                new_rem_t = rem_t - cost

                new_nu_indices = np.empty(len(nu_indices) + 1, dtype=np.int64)
                new_nu_values = np.empty(len(nu_values) + 1, dtype=np.int64)

                for ell in range(len(nu_indices)):
                    new_nu_indices[ell] = nu_indices[ell]
                    new_nu_values[ell] = nu_values[ell]

                new_nu_indices[len(nu_indices)] = dim_i
                new_nu_values[len(nu_values)] = j

                stack.append(
                    (
                        dim_i + 1,
                        new_rem_t,
                        new_parity,
                        new_prod_n,
                        new_nu_indices,
                        new_nu_values,
                    )
                )

                j += 1

    return total


def non_zero_indices_and_zetas(
    k: Sequence[float],
    t: float,
    quadrature: bool = False,
) -> tuple[defaultdict[int, list[tuple[tuple[int, int], ...]]], defaultdict[int, list[float]]]:
    r"""
    Computes the subset of multi-indices $\boldsymbol{\nu}$ in $\Lambda_{\boldsymbol{k}, t}$ that have non-zero Smolyak
    coefficient $\zeta_{\Lambda_{\boldsymbol{k},t}, \boldsymbol{\nu}}$, as well as these Smolyak coefficients, grouped
    by the sparsity level $n$ (number of non-zero entries in the multi-index $\boldsymbol{\nu}$).

    Parameters
    ----------
    k : Sequence[float]
        Weight vector of the anisotropy of the multi-index set. The dimension $d$ is inferred as `len(k)`.
    t : float
        Threshold parameter to control the cardinality of the multi-index set.

    Returns
    -------
    n2nus : defaultdict[int, list[tuple[tuple[int, int], ...]]]
        Dictionary mapping sparsity level $n$ to the list of multi-indices $\boldsymbol{\nu}$
        with $n$ non-zero entries and non-zero Smolyak coefficients.
    n2zetas : defaultdict[int, list[float]]
        Dictionary mapping sparsity level $n$ to the list of corresponding non-zero Smolyak coefficients
        $\zeta_{\Lambda_{\boldsymbol{k}, t}, \boldsymbol{\nu}}$.
    """
    d = len(k)
    n2nus, n2zetas = defaultdict(list), defaultdict(list)

    def cost_multiplier(j: int) -> int:
        return 2 if quadrature and j == 1 else j

    min_positive_multiplier = 2 if quadrature else 1

    stack = [(0, t, ())]
    while stack:
        i, rem_t, nu = stack.pop()
        # terminal skip check
        has_later_positive_extension = i + 1 < d and min_positive_multiplier * k[i + 1] < rem_t

        if i >= d or not has_later_positive_extension:
            if quadrature:
                nu_indices = np.array([idx for idx, _ in nu], dtype=np.int64)
                nu_values = np.array([val for _, val in nu], dtype=np.int64)
                zeta = smolyak_coefficient_quadrature(k, d, rem_t, 0, nu_indices, nu_values)
            else:
                zeta = smolyak_coefficient(k, d, rem_t, 0)
            if zeta != 0:
                n = len(nu)
                n2nus[n].append(nu)
                n2zetas[n].append(zeta)
        # expand exactly like indexset
        if i < d:
            if i + 1 < d and min_positive_multiplier * k[i + 1] < rem_t:
                stack.append((i + 1, rem_t, nu))
            j = 1
            while cost_multiplier(j) * k[i] < rem_t:
                stack.append((i + 1, rem_t - cost_multiplier(j) * k[i], nu + ((i, j),)))
                j += 1
    return n2nus, n2zetas


def nodeset_cardinality(k: Sequence[float], t: float, nested: bool = False, quadrature: bool = False) -> int:
    r"""
    Compute the cardinality of the set of interpolation nodes associated with the multi-index set
    $\Lambda_{\boldsymbol{k}, t}$.

    Parameters
    ----------
    k : Sequence[float]
        Weight vector of the anisotropy of the multi-index set. The dimension $d$ is inferred as `len(k)`.
    t : float
        Threshold parameter to control the cardinality of the multi-index set.
    nested : bool
        Boolean flag specifying whether the sequence of interpolation nodes used is nested or not.
    quadrature : bool, optional
        If True, the cardinality is computed for quadrature purposes.

    Returns
    -------
    int
        Cardinality of the set of interpolation nodes specified by $\Lambda_{\boldsymbol{k}, t}$.

    Notes
    -----
    * If `nested = True`, then the cardinality of the set of interpolation nodes is equal to the cardinality of
        $\Lambda_{\boldsymbol{k}, t}.$
    """
    if nested:
        return __nodeset_cardinality_nested(k, t, quadrature=quadrature)
    if quadrature:
        return __nodeset_cardinality_non_nested_quadrature(k, t)
    return __nodeset_cardinality_non_nested(k, t)


def find_approximate_threshold(
    k: Sequence[float], m: int, nested: bool, quadrature: bool = False, max_iter: int = 32, accuracy: float = 0.001
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
    while nodeset_cardinality(k, l_interval[0], nested, quadrature) > m:
        l_interval = [l_interval[0] / 1.2, l_interval[0]]
    while nodeset_cardinality(k, l_interval[1], nested, quadrature) < m:
        l_interval = [l_interval[1], l_interval[1] * 1.2]

    # bisect search interval
    def midpoint(interval):
        return interval[0] + (interval[1] - interval[0]) / 2.0

    t_cand = midpoint(l_interval)
    m_cand = nodeset_cardinality(k, t_cand, nested, quadrature)
    for _ in range(max_iter):
        if m_cand > m:
            l_interval = [l_interval[0], t_cand]
        else:
            l_interval = [t_cand, l_interval[1]]
        t_cand = midpoint(l_interval)
        m_cand = nodeset_cardinality(k, t_cand, nested, quadrature)

        if np.abs(m_cand - m) / m < accuracy:
            break

    return t_cand
