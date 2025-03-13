import numpy as np


# deprecated
def indexset_dense(k, l, idx=None) :
    if idx is None : idx = ()
    if len(k) == 0 :
        return [idx]
    r = []
    j = 0
    while j*k[0] < l :
        r += indexset_dense(k[1:], l - j * k[0], idx + (j,))
        j += 1
    return r


# deprecated
def unitball(nu, k, l, e=None) :
    if e is None : e = []
    if len(k) == 0 :
        return [e]
    r = unitball(nu[1:], k[1:], l-nu[0]*k[0], e + [0])
    if np.dot(nu,k) + k[0] < l :
        r += unitball(nu[1:], k[1:], l-(nu[0]+1)*k[0], e + [1])
    return r


def smolyak_coefficient_zeta_dense(k, l, *, nu=None) :
    return  np.sum([(-1) ** (np.sum(e)) for e in unitball(nu, k, l)])


def indexset_sparse(k, l, i=0, idx=None, *, cutoff=None) :
    if idx is None : idx = {}
    if cutoff is not None and i >= cutoff : return [idx]
    r = []
    if (cutoff is None or i+1 < cutoff) and k(i+1) < l :
        r += indexset_sparse(k, l, i+1, idx, cutoff=cutoff)
    else :
        r += [idx]
    j = 1
    while j * k(i) < l :
        r += indexset_sparse(k, l-j*k(i), i+1, {**idx, i : j}, cutoff=cutoff)
        j += 1
    return r


def abs_e_sparse(k, l, i=0, e=None, *, nu=None, cutoff=None) :
    if e is None :
        assert i == 0 and nu is not None
        e = 0
        l -= np.sum([nu[j] * k(j) for j in nu.keys()])
    if cutoff is not None and i >= cutoff :
        return [e]
    r = []
    if (cutoff is None or i+1 < cutoff) and k(i+1) < l :
        r += abs_e_sparse(k, l, i+1, e, cutoff=cutoff)
    else :
        r += [e]
    if k(i) < l :
        r += abs_e_sparse(k, l-k(i), i+1, e+1, cutoff=cutoff)
    return r


def smolyak_coefficient_zeta_sparse(k, l, *, nu=None, cutoff=None) :
    return np.sum([(-1)**e for e in abs_e_sparse(k, l, nu=nu, cutoff=cutoff)])


def sparse_index_to_dense(nu, cutoff=None) :
    if cutoff is None : cutoff = max(nu.keys())
    nnu = [0] * cutoff
    for k, v in nu.items() :
        nnu[k] = v
    return tuple(nnu)


def dense_index_to_sparse(nu) :
    nnu = {}
    for k, v in enumerate(nu) :
        if v > 0 : nnu[k] = v
    return nnu


def n_points(kmap, l, cutoff, nested=False) :
    iset = indexset_sparse(kmap, l, cutoff=cutoff)

    if nested :
        return len(iset)

    n = 0
    for nu in iset :
        c = np.sum([(-1)**e for e in abs_e_sparse(kmap, l, nu=nu, cutoff=cutoff)])
        if c != 0 :
            n += np.prod([v+1 for v in nu.values()])
    return n


def find_suitable_l(k, n=50, nested=False) :
    assert n > 0
    kmap = lambda j : k[j]
    cutoff = len(k)

    if n == 1 : return 1

    # establish search interval
    l_interval = [1, 2]
    while n_points(kmap, l_interval[0], cutoff, nested) > n :
        l_interval[0] /= 1.2
    while n_points(kmap, l_interval[1], cutoff, nested) < n :
        l_interval[1] *= 1.2

    # bisect search interval
    midpoint = lambda interval : interval[0] + (interval[1] - interval[0])/2
    l_cand = midpoint(l_interval)
    n_cand = n_points(kmap, l_cand, cutoff, nested)
    for _ in range(32) :
        if n_cand > n : l_interval = [l_interval[0], l_cand]
        else :          l_interval = [l_cand, l_interval[1]]
        l_cand = midpoint(l_interval)
        n_cand = n_points(kmap, l_cand, cutoff, nested)
        if n_cand == n : break
    return l_cand
