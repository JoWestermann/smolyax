import itertools as it

import numpy as np
import setup

from jax_smolyak import indices
from jax_smolyak.tensorproduct import TensorProductBarycentricInterpolator


def test_tensorproduct_interpolation():
    for node_gen in setup.generate_nodes(n=10, dmin=1, dmax=4):
        nu = np.random.randint(low=1, high=10, size=node_gen.dim)
        f = setup.generate_test_function_tensorproduct(node_gen=node_gen, nu=nu)
        ip = TensorProductBarycentricInterpolator(node_gen, indices.dense_index_to_sparse(nu), node_gen.dim, f)

        print(f"Testing with nu = {nu},", node_gen)
        print("\t ... testing interpolation points")
        for x in it.product(*ip.nodes):
            x = np.array(x)
            assert np.isclose(f(x), ip(x)), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"

        print("\t ... testing random points")
        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(f(x), ip(x)), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"
