import sys
from functools import lru_cache

sys.setrecursionlimit(11_000)


def indexset_original(k, t, i=0, nu=None):
    if nu is None:
        nu = {}
    if i >= len(k):
        return [nu]
    result = []
    if i + 1 < len(k) and k[i + 1] < t:
        result += indexset_original(k, t, i + 1, nu)
    else:
        result.append(nu)
    j = 1
    while j * k[i] < t:
        new_nu = nu.copy()
        new_nu[i] = j
        result += indexset_original(k, t - j * k[i], i + 1, new_nu)
        j += 1
    return result


def indexset_lru(k, t):
    k_list = tuple(k)
    n = len(k_list)

    @lru_cache(maxsize=None)
    def recurse(i, remaining):
        if i >= n:
            return [dict()]
        result = []
        if i + 1 < n and k_list[i + 1] < remaining:
            result += recurse(i + 1, remaining)
        else:
            result.append(dict())

        j = 1
        while j * k_list[i] < remaining:
            for nu in recurse(i + 1, remaining - j * k_list[i]):
                new_nu = nu.copy()
                new_nu[i] = j
                result.append(new_nu)
            j += 1
        return result

    return recurse(0, t)


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



if __name__ == "__main__":
    import time
    import numpy as np
    import random

    print(f" size   || original ||    tuples    || ")

    total_original = []
    total_tuple = []
    for _ in range(20):
        dim = random.randint(10, 1000)
        a = random.uniform(1.1, 5)
        b = random.uniform(0.1, 5)
        k = [np.log(a + i * b) for i in range(dim)]
        t = 10 * np.mean(k) / np.log(dim)

        start = time.perf_counter()
        for _ in range(10) :
            c_original = len(indexset_original(k, t))
        t_original = time.perf_counter() - start
        total_original += [t_original]


        start = time.perf_counter()
        for _ in range(10) :
            c_tuple = len(indexset_tuples(k, t))
        t_tuple = time.perf_counter() - start
        total_tuple += [t_tuple]


        if not c_original == c_tuple :
            print('Unequal sets!')
            print('k = ', k)
            print('t = ', t)

        print(f"{c_original:<7} ||  {t_original:.4f}  || {t_tuple:.4f} / {t_tuple/t_original:.2f} ||") # || {t_lru:.4f} / {t_lru/t_original:.2f}

    rel_times = np.array(total_tuple) / np.array(total_original)
    print(f'\n {np.mean(rel_times):.3f} +- {np.std(rel_times):.3f}')

