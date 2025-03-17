import setup

from jax_smolyak.tensorproduct import *


def test_tensorproduct_interpolation():
    for g in setup.generate_pointsets(n=10, dmin=1, dmax=4):
        nu = np.random.randint(low=1, high=10, size=g.d)
        f = setup.generate_test_function_tensorproduct(g=g, nu=nu)
        ip = TensorProductBarycentricInterpolator(
            g, indices.dense_index_to_sparse(nu), g.d, f
        )

        print(f"Testing with nu = {nu},", g)
        print("\t ... testing interpolation points")
        for x in it.product(*ip.nodes):
            x = np.array(x)
            assert np.isclose(
                f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"

        print("\t ... testing random points")
        for n in range(5):
            x = g.get_random(n=np.random.randint(low=0, high=5))
            assert np.allclose(
                f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"
