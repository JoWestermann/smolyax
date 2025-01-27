import numpy as np

from points import *

def generate_pointsets(n, dmax) :

    sets = []
    for _ in range(n) :

        d = np.random.randint(low=2, high=dmax)

        m = np.random.randn(d)
        a = np.random.rand(d)
        sets.append(GaussHermiteMulti(m, a))

        domain = np.zeros((d, 2))
        domain[:,1] = np.sort(np.random.rand(d)*10)
        domain[:,0] = - domain[:,1]
        sets.append(LejaMulti(domains=domain))

    return sets


def test_pointsets() :

    for g in generate_pointsets(5, 10) :
        x = g.get_random()
        assert np.isclose(x, g.scale(g.scale_back(x))).all()

