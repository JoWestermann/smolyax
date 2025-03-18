import numpy as np
import setup


def test_pointsets():

    for node_gen in setup.generate_nodes(n=5, dmin=2, dmax=10):
        n = np.random.randint(low=0, high=5)
        x = node_gen.get_random(n)

        if n == 0:
            assert x.shape == (node_gen.dim,)
        else:
            assert x.shape == (n, node_gen.dim)
        assert np.allclose(x, node_gen.scale(node_gen.scale_back(x)))
