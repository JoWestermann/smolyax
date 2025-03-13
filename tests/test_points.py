import numpy as np

from jax_smolyak.points import *

import setup


def test_pointsets() :

    for g in setup.generate_pointsets(5, 10) :
        x = g.get_random()
        assert np.isclose(x, g.scale(g.scale_back(x))).all()
