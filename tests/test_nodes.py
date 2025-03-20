import numpy as np
import setup


def test_nodes_scaling():

    # test with default constructors
    for node_gen in setup.generate_nodes_default(dmin=1, dmax=3):
        n = np.random.randint(low=1, high=5)
        x = node_gen.get_random(n)

        assert x.shape == (n, node_gen.dim)
        assert np.allclose(x, node_gen.scale(node_gen.scale_back(x)))

    # test with specific domains
    for node_gen in setup.generate_nodes(n=5, dmin=2, dmax=10):
        n = np.random.randint(low=0, high=5)
        x = node_gen.get_random(n)

        if n == 0:
            assert x.shape == (node_gen.dim,)
        else:
            assert x.shape == (n, node_gen.dim)
        assert np.allclose(x, node_gen.scale(node_gen.scale_back(x)))
