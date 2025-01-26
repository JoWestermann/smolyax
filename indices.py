import numpy as np

# deprecated
def indexset(k, l, idx=None) :
    if idx is None : idx = []
    if len(k) == 0 :
        return [idx]
    r = []
    j = 0
    while(j*k[0] < l) :
        r += indexset(k[1:], l-j*k[0], idx + [j])
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

def indexset_sparse(k, l, i=0, idx=None, *, cutoff=None) :
    if idx is None : idx = {}
    if cutoff is not None and i >= cutoff : return [idx]
    r = []
    if (cutoff is None or i+1 < cutoff) and k(i+1) < l :
        r += indexset_sparse(k, l, i+1, idx, cutoff=cutoff)
    else :
        r += [idx]
    j = 1
    while(j * k(i) < l) :
        r += indexset_sparse(k, l-j*k(i), i+1, {**idx, i : j}, cutoff=cutoff)
        j += 1
    return r

def abs_e_sparse(k, l, i=0, e=None, *, nu=None, cutoff=None) :
    if e is None :
        assert(i == 0 and nu is not None)
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

def sparse_index_to_dense(nu, d=None):
    if d is None : d = max(nu.keys())
    nnu = [0] * d
    for key, value in nu.items():
        nnu[key] = value
    return tuple(nnu)

def dense_index_to_sparse(nu):
    nnu = {}
    for k,v in nu :
        if v > 0 : nnu[k] = v
    return nnu

if __name__ == '__main__' :

    import itertools as it

    if False :
        for _ in range(10) :
            d = np.random.randint(low=1, high=5)
            k = sorted(np.random.randint(low=1, high=10, size=d))
            k /= k[0]
            n = np.random.randint(low=1, high=130)
            find_suitable_l(k, n)

    def compare_sparse_dense(isparse, idense, i=0) :
        if i >= len(idense[0]) : return True
        assert len(isparse) == len(idense), f'{len(isparse)} {len(idense)}'
        dsparse = {}
        for nu in isparse :
            key = 0 if i not in nu.keys() else nu[i]
            if key in dsparse.keys() :
                dsparse[key].append(nu)
            else :
                dsparse[key] = [key]
        ddense = {}
        for nu in idense :
            key = nu[i]
            if key in ddense.keys() :
                ddense[key].append(nu)
            else :
                ddense[key] = [key]
        assert(dsparse.keys() == ddense.keys())
        for key in dsparse.keys() :
            compare_sparse_dense(isparse, idense, i=i+1)

    for _ in range(10) :

        d = np.random.randint(low=1, high=5)
        k = sorted(np.random.randint(low=1, high=10, size=d))
        k /= k[0]
        l = np.random.rand() * np.random.randint(low=1, high=10)
        i = indexset_sparse(lambda j : k[j], l, cutoff=d)
        ii = indexset(k, l)

        # test 1
        compare_sparse_dense(i, ii)

        # test 2
        for idx in it.product(*[range(int(np.floor(ki))+2) for ki in k]) :
            idx_sparse = {k: v for k, v in enumerate(idx) if v > 0}
            assert (idx_sparse in i) == (np.dot(idx, k) < l), \
                    f"Assertion failed with\n k = {k}, l = {l},\n idx = {idx},\n idx*k = {np.dot(idx, k)}, " + \
                    f"\n (idx in i) = {idx in i},\n np.dot(idx, k) < l = {np.dot(idx, k) < l}"
            assert (list(idx) in ii) == (np.dot(idx, k) < l), \
                    f"Assertion failed with\n k = {k}, l = {l},\n idx = {idx},\n idx*k = {np.dot(idx, k)}, " + \
                    f"\n (idx in i) = {idx in i},\n np.dot(idx, k) < l = {np.dot(idx, k) < l}"

    print('TEST indexset and indexset == indexset_sparse SUCCESSFUL')


    d0 = int(np.ceil(l/k[0]))
    d1 = int(np.ceil(l/k[1]))
    arr = [[' ' for _ in range(int(np.ceil(l/k[1])))] for _ in range(d0)]
    i = indexset(k, l)
    print('\n', i, '\n')
    for idx in i :
        c = np.sum([(-1)**(np.sum(e)) for e in unitball(idx, k, l)])
        if c == 0 :
            arr[idx[0]][idx[1]] = '0'
        elif c == 1 :
            arr[idx[0]][idx[1]] = '+'
        elif c == -1 :
            arr[idx[0]][idx[1]] = '-'

    for i in range(d0)[::-1]  :
        print('     +' + '---+'*len(arr[i]) + '\n {:3} | '.format(i), end='')
        for j in range(len(arr[i])) :
            print(arr[i][j] + ' | ', end='')
        print()
    print('     +' + '---+'*len(arr[-1]))
    print('       ', end='')
    for i in range(d1) : print(str(i) + '   ', end='')
    print()

