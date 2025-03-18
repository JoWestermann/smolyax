import numpy as np
import setup

from jax_smolyak.smolyak_jax import MultivariateSmolyakBarycentricInterpolator


def test_smolyak_jax():
    print("\nTesting vector-valued Smolyak operator (jax) ...")

    for g in setup.generate_pointsets(n=10, dmin=1, dmax=4):

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        d_out = np.random.randint(low=1, high=4)
        t = np.random.randint(low=1, high=4)
        print(f"... with k = {k}, t = {t}, d_out = {d_out},", g)

        f = setup.generate_test_function_smolyak(g=g, k=k, t=t, d_out=d_out)

        ip = MultivariateSmolyakBarycentricInterpolator(
            g=g, k=k, t=t, d_out=d_out, batchsize=1, f=f
        )

        for n in range(5):
            x = g.get_random(n=np.random.randint(low=0, high=5))
            assert np.allclose(
                f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"
