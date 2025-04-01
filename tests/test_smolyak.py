import numpy as np
import setup

from jax_smolyak.smolyak import MultivariateSmolyakBarycentricInterpolator


def test_smolyak_jax():
    print("\nTesting vector-valued Smolyak operator (jax) ...")

    for node_gen in setup.generate_nodes(n=20, dmin=1, dmax=4):

        k = sorted(np.random.uniform(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        d_out = np.random.randint(low=1, high=4)
        t = np.random.uniform(low=1, high=8)
        print(f"... with k = {k}, t = {t}, d_out = {d_out},", node_gen)

        f = setup.generate_test_function_smolyak(node_gen=node_gen, k=k, t=t, d_out=d_out)

        ip = MultivariateSmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=d_out, batchsize=1, f=f)

        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(
                f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"
