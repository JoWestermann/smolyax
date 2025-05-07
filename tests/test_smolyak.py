import numpy as np
import jax
import setup

from jax_smolyak.smolyak import SmolyakBarycentricInterpolator


def test_smolyak_on_polynomials():
    print("\nTesting that the Smolyak operator interpolates suitable polynomials exactly.")

    for node_gen in setup.generate_nodes(n=20, dmin=1, dmax=4):

        k = sorted(np.random.uniform(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        d_out = np.random.randint(low=1, high=4)
        t = np.random.uniform(low=1, high=8)
        print(f"... with k = {k}, t = {t}, d_out = {d_out},", node_gen)

        f = setup.generate_test_function_smolyak(node_gen=node_gen, k=k, t=t, d_out=d_out)

        ip = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=d_out, batchsize=1, f=f)

        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(
                f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"


def test_smolyak_target_function():
    print("\nCheck that the Smolyak operator can work with numpy or jax target functions.")

    for node_gen in setup.generate_nodes(n=1, dmin=3, dmax=3):

        k = sorted(np.random.uniform(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        t = np.random.uniform(low=2, high=3)
        print(f"... with k = {k}, t = {t}", node_gen)

        # test numpy

        def test_f(x):
            return (x + 1000) / (x + 1000)

        ip = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=len(k), batchsize=1, f=test_f)

        for n in range(3):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(
                test_f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {test_f(x)}\n ip(x) = {ip(x)} @ n = {n}"

        # test jax

        test_f_jax_cpu = jax.jit(test_f)

        ip = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=len(k), batchsize=1, f=test_f_jax_cpu)

        for n in range(3):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(
                test_f_jax_cpu(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {test_f_jax_cpu(x)}\n ip(x) = {ip(x)} @ n = {n}"
