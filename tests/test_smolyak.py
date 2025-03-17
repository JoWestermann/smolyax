import numpy as np
import setup
import test_tensorproduct

from jax_smolyak.smolyak import *


def generate_test_function_univariate(ip, n=1):
    coeffs = np.random.rand(n) * 2 - 1
    coeffs /= len(coeffs)
    idxs = np.random.randint(low=0, high=len(ip.operators), size=n)
    fs = [test_tensorproduct.generate_test_function(ip.operators[i]) for i in idxs]

    def test_f(x, coeffs, fs):
        res = 0
        for c, fi in zip(coeffs, fs):
            res += c * fi(x)
        return res

    return lambda x: test_f(x, coeffs, fs)


def generate_test_function_multivariate(ip):
    fs = [generate_test_function_univariate(c) for c in ip.components]
    return lambda x: np.array([fi(x) for fi in fs])


def test_smolyak_univariate():

    for g in setup.generate_pointsets(n=10, dmin=2, dmax=2):

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        print(f"Testing with d = {g.d}, k = {k}")
        # g.print()

        ip = SmolyakBarycentricInterpolator(g, k, 2)
        f = generate_test_function_univariate(ip)
        ip.set_F(f)

        for n in range(5):
            x = g.get_random()
            print(f"\t\t ip(x) = {ip(x)}, f(x) = {f(x)}")
            assert np.isclose(
                ip(x), f(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"


def test_smolyak_multivariate():

    for g in setup.generate_pointsets(n=10, dmin=2, dmax=5):

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        d2 = np.random.randint(low=1, high=5)
        k2 = sorted(np.random.randint(low=1, high=4, size=d2), reverse=True)
        print(f"Testing with d2 = {d2}, k2 = {k2}")

        ip = MultivariateSmolyakBarycentricInterpolator(g=g, k=k, l=k2)
        f = generate_test_function_multivariate(ip)
        ip.set_F(f=f)

        for n in range(5):
            x = g.get_random()
            print(f"\t\t ip(x) = {ip(x)}, f(x) = {f(x)}")
            assert np.isclose(
                ip(x), f(x)
            ).all(), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"
