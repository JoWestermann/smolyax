import numpy as np
from numpy.typing import ArrayLike


def indexset(k, t: float, dim_i: int = 0, nu: dict[int, int] = None):
    if nu is None:
        nu = {}
    if dim_i >= len(k):
        return [nu]
    r = []
    if dim_i + 1 < len(k) and k[dim_i + 1] < t:
        r += indexset(k, t, dim_i + 1, nu)
    else:
        r += [nu]
    j = 1
    while j * k[dim_i] < t:
        r += indexset(k, t - j * k[dim_i], dim_i + 1, {**nu, dim_i: j})
        j += 1
    return r


def indexset_tuples(k, t):
    stack = [(0, t, ())]
    result = set()

    while stack:
        dim_i, remaining_t, nu = stack.pop()

        if dim_i >= len(k):
            result.add(nu)
            continue

        # Case 1: Try skipping this dimension
        if dim_i + 1 < len(k) and k[dim_i + 1] < remaining_t:
            stack.append((dim_i + 1, remaining_t, nu))
        else:
            result.add(nu)

        # Case 2: Try all j ≥ 1 while feasible
        j = 1
        while j * k[dim_i] < remaining_t:
            # Create new sparse index
            nu_extended = tuple(list(nu) + [(dim_i, j)])
            new_t = remaining_t - j * k[dim_i]
            stack.append((dim_i + 1, new_t, nu_extended))
            j += 1

    return result

def count_indexset_tuples(k, t):
    stack = [(0, t, ())]
    count = 0

    while stack:
        dim_i, remaining_t, nu = stack.pop()

        if dim_i >= len(k):
            count +=1
            continue

        # Case 1: Try skipping this dimension
        if dim_i + 1 < len(k) and k[dim_i + 1] < remaining_t:
            stack.append((dim_i + 1, remaining_t, nu))
        else:
            count+=1

        # Case 2: Try all j ≥ 1 while feasible
        j = 1
        while j * k[dim_i] < remaining_t:
            # Create new sparse index
            nu_extended = tuple(list(nu) + [(dim_i, j)])
            new_t = remaining_t - j * k[dim_i]
            stack.append((dim_i + 1, new_t, nu_extended))
            j += 1

    return count


def abs_e_tuple_nu(k, t, i=0, e=None, *, nu: tuple = None):
    if e is None:
        assert i == 0 and nu is not None
        e = 0
        t -= np.sum([nu_j * k[j] for j, nu_j in nu])
    if i >= len(k):
        return [e]
    r = []
    if i + 1 < len(k) and k[i + 1] < t:
        r += abs_e_tuple_nu(k, t, i + 1, e)
    else:
        r += [e]
    if k[i] < t:
        r += abs_e_tuple_nu(k, t - k[i], i + 1, e + 1)
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


def smolyak_coefficient_zeta(k, t, i=0, e=0, *, nu: dict[int, int] = None):
    if nu is None:
        nu = {}
    if i == 0 and nu is not None:
        t -= sum(nu[j] * k[j] for j in nu)
    # Base case: if we've processed all dimensions, return the contribution (-1)**e.
    if i >= len(k):
        return 1 - 2 * (e & 1)
    coeff = 0
    # If the next k (if available) is less than t, we continue recursively;
    # otherwise we add the current contribution.
    if i + 1 < len(k) and k[i + 1] < t:
        coeff += smolyak_coefficient_zeta(k, t, i + 1, e, nu=nu)
    else:
        coeff += 1 - 2 * (e & 1)
    # If k[i] is less than t, subtract it from t and increase e (flipping the sign).
    if k[i] < t:
        coeff += smolyak_coefficient_zeta(k, t - k[i], i + 1, e + 1, nu=nu)
    return coeff


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
    if nested:
        return count_indexset_tuples(k,t)
    else:
        iset = indexset_tuples(k, t)

    n = 0
    for nu in iset:
        c = np.sum([(-1) ** e for e in abs_e_tuple_nu(k, t, nu=nu)])
        if c != 0:
            n += np.prod([v + 1 for _, v in nu])
    return n


def find_suitable_t(k: ArrayLike, m: int = 50, nested: bool = False, max_iter=32, accuracy=0.001) -> int:
    """
    k : weight vector of the anisotropy of the multi-index set
    m : target cardinality of the set of interpolation nodes
    nested : flag to indicate whether nested or non-nested interpolation nodes are used
    max_iter : maximal number of bisection iterations
    accuracy : relative tolerance within which the cardinality of the set of interpolation nodes may deviate from m
    returns t : threshold parameter to construct a k-weighted multi-index set of size (roughly) m
    """
    assert m > 0

    if m == 1:
        return 1

    # establish search interval
    l_interval = [1, 2]
    while cardinality(k, l_interval[0], nested) > m:
        l_interval = [l_interval[0] / 1.2, l_interval[0]]
    while cardinality(k, l_interval[1], nested) < m:
        l_interval = [l_interval[1], l_interval[1] * 1.2]

    # bisect search interval
    def midpoint(interval):
        return interval[0] + (interval[1] - interval[0]) / 2

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
