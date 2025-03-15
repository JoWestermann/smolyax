import numpy as np
import setup
import test_smolyak

from jax_smolyak.smolyak import MultivariateSmolyakBarycentricInterpolator as SmolN
from jax_smolyak.smolyak_jax import MultivariateSmolyakBarycentricInterpolator as SmolJ


def test_smolyak_jax():

    for g in setup.generate_pointsets(10, 3):

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        d2 = np.random.randint(low=1, high=4)
        k2 = np.random.randint(low=1, high=4)
        print(f"Testing with d2 = {d2}, k2 = {k2}")

        ipN = SmolN(g=g, k=k, l=[k2] * d2)
        f = test_smolyak.generate_test_function_multivariate(ipN)

        ipJ = SmolJ(g=g, k=k, l=k2, rank=d2, batchsize=1, f=f)

        for n in range(5):
            x = g.get_random()
            y = f(x)
            y_ = ipJ(x)
            assert np.isclose(
                y, y_
            ).all(), (
                f"Assertion failed with\n x = {x}\n f(x) = {y}\n ip(x) = {y_} @ n = {n}"
            )
