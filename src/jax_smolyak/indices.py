import numpy as np
from collections import deque
import numpy as np

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


def indexset_sparse(k, l, cutoff, i=0, idx=None):
    if idx is None:
        idx = cykhash.Int32toInt32Map()
    if i >= cutoff:
        return [idx]
    r = []
    if (i + 1 < cutoff) and k(i + 1) < l:
        r += indexset_sparse(k, l, cutoff, i + 1, idx)
    else:
        r += [idx]
    j = 1
    while j * k(i) < l:
        if i not in idx:  # Only allow `i: j` if it hasn't been assigned
            idx[i] = j  # Assign `i: j`
            r += indexset_sparse(k, l - j * k(i),cutoff, i + 1, idx)
            del idx[i]  # Restore state after recursion
        j += 1
    return r

import cykhash
def indexset_sparse_count_fast_w_cutoff(k, l, cutoff, i=0, idx=None):
    """Optimized recursive count of sparse index sets while preventing redundancy."""
    
    if idx is None:
        idx = cykhash.Int32toInt32Map()

    # Base case: If depth is exceeded, count as valid
    if i >= cutoff:
        return 1

    count_val = 0

    # Case 1: Skip k(i) and move to the next index
    if (i + 1 < cutoff) and k(i + 1) < l:
        count_val += indexset_sparse_count_fast_w_cutoff(k, l, cutoff, i + 1, idx)
    else:
        count_val += 1

    # Case 2: Include multiples of k(i) and recurse
    j = 1
    while j * k(i) < l:
        if i not in idx:  # Only allow `i: j` if it hasn't been assigned
            idx[i] = j  # Assign `i: j`
            count_val += indexset_sparse_count_fast_w_cutoff(k, l - j * k(i),cutoff, i + 1, idx)
            del idx[i]  # Restore state after recursion
        j += 1

    return count_val


def abs_e_sparse(k, l, i=0, e=None, *, nu=None, cutoff=None): #speed me up!
    if e is None:
        assert i == 0 and nu is not None
        e = 0
        # print(nu)
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

#scratchwork. needs k to be an array, and need to think abuot sparse nu, maybe making it dense before this function so that the inner product is fast?
# @numba.njit(fastmath=True, cache=True)
def abs_e_sparse_fast_pow_w_cutoff_numba(k, l, cutoff, i=0, e=-1, nu=None):
    """Optimized version with bitwise operations, direct assignments, and proper handling of `nu`."""
    
    # Ensure `e` is properly initialized
    if e == -1:  # Equivalent to `if e is None` but works with numba
        e = 0
        if nu is not None:  # **Fix: Only compute `l -= sum(...)` if `nu` exists**
            l -= np.sum(np.array([nu[j] * k[j] for j in nu], dtype=np.int64))  

    # **Base Case: If `i >= cutoff`, compute (-1)^e directly**
    if i >= cutoff:
        return 1 - ((e & 1) << 1)  # Optimized (-1)^e using bit-shift

    i_plus_1 = i + 1  # Precompute to avoid redundant calculations

    # **Case 1: Skip `k[i]` and move to the next index**
    should_recurse_1 = (i_plus_1 < cutoff) & (k[i_plus_1] < l)  # Bitwise AND for branching avoidance

    count_val = (
        should_recurse_1 * abs_e_sparse_fast_pow_w_cutoff_numba(k, l, cutoff, i_plus_1, e, nu)  # Recursively call only when needed
        + (1 - ((e & 1) << 1)) * ~should_recurse_1  # Compute (-1)^e if recursion is skipped
    )

    # **Case 2: Include `k[i]` and recurse**
    k_i = k[i] 
    should_recurse_2 = k_i < l  # Avoid unnecessary recursion

    count_val += should_recurse_2 * abs_e_sparse_fast_pow_w_cutoff_numba(k, l - k_i, cutoff, i_plus_1, e + 1, nu)

    return count_val

def abs_e_sparse_fast_pow_w_cutoff(k, l, cutoff, i=0, e=None, *, nu=None):
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
    if (i_plus_1< cutoff) and k(i_plus_1) < l:
        count_val += abs_e_sparse_fast_pow_w_cutoff(k, l, cutoff, i_plus_1, e)

    else:
        count_val += 1 - 2 * (e & 1)  # Inline computation of (-1)^e

    # Case 2: Include k(i) and recurse
    k_i = k(i)
    if k_i < l:
        count_val += abs_e_sparse_fast_pow_w_cutoff(k, l - k_i, cutoff, i_plus_1, e + 1)

    return count_val

def count_index_sets(k, l, cutoff=None):
    """
    Compute the number of valid index sets for given l using k(i), 
    without constructing them explicitly.
    """
    dp = np.zeros(l + 1, dtype=int)
    dp[0] = 1  # Base case: one way to sum to zero

    for i in range(cutoff or l):
        new_dp = dp.copy()
        j = 1
        while j * k(i) <= l:
            new_dp[j * k(i):] += dp[:-j * k(i)]
            j += 1
        dp = new_dp

    return dp[l]


def indexset_sparse_stack(k, l, *, cutoff=None):
    """
    Generate index sets efficiently using an iterative stack-based approach 
    with precomputed result size to avoid list resizing.
    """
    precomputed_size = count_index_sets(k, l, cutoff=cutoff)
    results = [None] * precomputed_size  # Preallocate space
    index = 0  # Tracking insertion point

    stack = deque([(0, {}, l)])  # Use deque for fast popping

    while stack:
        i, idx, l = stack.pop()

        if cutoff is not None and i >= cutoff:
            results[index] = idx
            index += 1
            continue

        # Push (i+1, idx, l) if condition allows
        if (cutoff is None or i + 1 < cutoff) and k(i + 1) < l:
            stack.append((i + 1, idx, l))
        else:
            results[index] = idx
            index += 1

        # Generate index sets efficiently
        j = 1
        while j * k(i) < l:
            new_idx = idx.copy()
            new_idx[i] = j
            stack.append((i + 1, new_idx, l - j * k(i)))
            j += 1

    return results[:index]  # Trim any unused slots


def count_abs_e_sparse(k, l, nu, cutoff=None):
    """
    Precompute the number of valid e values efficiently.
    """
    assert nu is not None, "nu must be provided"
    
    # Adjust `l`
    l -= np.sum([nu[j] * k(j) for j in nu.keys()])
    
    dp = np.zeros(l + 1, dtype=int)
    dp[0] = 1  # Base case: one way to sum to zero

    for i in range(cutoff or l):
        new_dp = dp.copy()
        if k(i) < l:
            new_dp[k(i):] += dp[:-k(i)]
        dp = new_dp

    return dp[l]  # Total number of valid e values


def abs_e_sparse_stack(k, l, *, nu=None, cutoff=None):
    """
    Compute the set of valid e values using an iterative approach 
    with a precomputed result size.
    """
    assert nu is not None, "nu must be provided"
    
    # Initialize `e` and adjust `l`
    e = 0
    l -= np.sum([nu[j] * k(j) for j in nu.keys()])

    # Precompute the expected size
    precomputed_size = count_abs_e_sparse(k, l, nu, cutoff) 
    results = np.empty(precomputed_size, dtype=int)  # Preallocate results array
    index = 0  # Index for inserting results

    # Use deque for faster stack handling
    stack = deque([(0, e, l)])  # (i, e, l)

    while stack:
        i, e, l = stack.pop()

        if cutoff is not None and i >= cutoff:
            results[index] = e
            index += 1
            continue

        # Case when k(i) < l
        if k(i) < l:
            stack.append((i + 1, e + 1, l - k(i)))

        # Case when k(i+1) < l
        if (cutoff is None or i + 1 < cutoff) and k(i + 1) < l:
            stack.append((i + 1, e, l))
        else:
            results[index] = e
            index += 1

    return results[:index]  # Trim unused slots


def smolyak_coefficient_zeta_sparse(k, l, *, nu=None, cutoff=None):
    # real = np.sum(1 - 2 * (np.array(abs_e_sparse(k, l, nu=nu, cutoff=cutoff)) & 1))
    # print(real)
    # fake = abs_e_sparse_fast_pow(k, l, nu=nu, cutoff=cutoff)
    # print("fake",fake, "\n")
    return  abs_e_sparse_fast_pow_w_cutoff(k, l, cutoff, nu=nu)


def sparse_index_to_dense(nu, cutoff=None):
    if cutoff is None:
        cutoff = max(nu.keys())
    nnu = [0] * cutoff
    for k, v in nu.items():
        nnu[k] = v
    return tuple(nnu)


def dense_index_to_sparse(nu):
    nnu = {}
    for k, v in enumerate(nu):
        if v > 0:
            nnu[k] = v
    return nnu


def n_points(kmap, l, cutoff, nested=False):
    # print("computing n_points")
    # iset = indexset_sparse(kmap, l, cutoff=cutoff)
    # assert all(indexset_sparse(kmap, l, cutoff=cutoff) == indexset_sparse_stack(kmap, l, cutoff=cutoff))
    if nested:
        # iset = indexset_sparse(kmap, l, cutoff=cutoff)
        # print(indexset_sparse_count_fast(kmap, l, cutoff=cutoff))
        # try:
        #     print("other",indexset_sparse_count(kmap, l, cutoff=cutoff))
        # except:
        #     pass
        return indexset_sparse_count_fast_w_cutoff(kmap, l, cutoff) #len(iset)
        
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


def find_suitable_l(k, n=50, nested=False):
    assert n > 0
    kmap = lambda j: k[j]
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
    midpoint = lambda interval: interval[0] + (interval[1] - interval[0]) / 2
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
