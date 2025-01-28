import numpy as np

from points import *



def test_pointsets() :

    for g in generate_pointsets(5, 10) :
        x = g.get_random()
        assert np.isclose(x, g.scale(g.scale_back(x))).all()

