import numpy as np
import setup

from jax_smolyak.smolyak import (MultivariateSmolyakBarycentricInterpolator,
                                 SmolyakBarycentricInterpolator)


def test_smolyak_scalar():
    print("\nTesting scalar-valued Smolyak operator (numpy) ...")

    for node_gen in setup.generate_nodes(n=10, dmin=1, dmax=4):

        k = sorted(np.random.randint(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        t = np.random.randint(low=1, high=4)
        print(f"... with k = {k}, t = {t},", node_gen)

        ip = SmolyakBarycentricInterpolator(node_gen, k, t)
        ff = setup.generate_test_function_smolyak(node_gen=node_gen, k=k, t=t, d_out=1)

        def f(x):
            return np.squeeze(ff(x))

        ip.set_f(f)

        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(ip(x), f(x)), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"


def test_smolyak_vector():
    print("\nTesting vector-valued Smolyak operator (numpy) ...")

    for node_gen in setup.generate_nodes(n=10, dmin=1, dmax=4):

        k = sorted(np.random.randint(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        d_out = np.random.randint(low=1, high=5)
        t = sorted(np.random.randint(low=1, high=4, size=d_out), reverse=True)
        print(f"... with k = {k}, t = {np.array(t).tolist()},", node_gen)

        ip = MultivariateSmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t)
        f = setup.generate_test_function_smolyak(node_gen=node_gen, k=k, t=t, d_out=d_out)
        ip.set_f(f=f)

        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            print(f"\t\t ip(x) = {ip(x)}, f(x) = {f(x)}")
            assert np.allclose(
                ip(x), f(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"
