import numpy as np


def indexset(k, t, idx=None):
    if idx is None:
        idx = ()
    if len(k) == 0:
        return [idx]
    r = []
    j = 0
    while j * k[0] < t:
        r += indexset(k[1:], t - j * k[0], idx + (j,))
        j += 1
    return r


def unitball(nu, k, t, e=None):
    if e is None:
        e = []
    if len(k) == 0:
        return [e]
    r = unitball(nu[1:], k[1:], t - nu[0] * k[0], e + [0])
    if np.dot(nu, k) + k[0] < t:
        r += unitball(nu[1:], k[1:], t - (nu[0] + 1) * k[0], e + [1])
    return r


def smolyak_coefficient(k, t, *, nu):
    return np.sum([(-1) ** (np.sum(e)) for e in unitball(nu, k, t)])
