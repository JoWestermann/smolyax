import numpy as np
import setup


def test_pointsets():

    for g in setup.generate_pointsets(n=5, dmin=2, dmax=10):
        x = g.get_random()
        assert np.allclose(x, g.scale(g.scale_back(x)))
