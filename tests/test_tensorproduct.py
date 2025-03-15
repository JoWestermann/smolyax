import itertools as it

import numpy as np
import scipy as sp
import setup

from jax_smolyak import indices, points
from jax_smolyak.tensorproduct import *


def generate_test_function(ip, n=1):
    # Generate a random polynomial that is interpolated exactly by the given tensor product interpolator
    coeffs = np.random.rand(n) * 2 - 1
    coeffs /= sum(coeffs)
    idxs = np.array(
        [np.random.randint(low=0, high=d + 1, size=n) for d in ip.degrees]
    ).T

    print(
        f"Generating random polynomial as a weighted sum of {n} basis polynomials in {len(ip.degrees)} dimensions.\n"
        + "Weights:\n",
        coeffs,
        "\nDegrees:\n",
        idxs,
    )

    if isinstance(ip.gens, points.LejaMulti):
        polys = [sp.special.legendre(d) for d in range(max(ip.degrees) + 1)]

        def test_f(x, coeffs, polys, degrees, gen):
            x = gen.scale_back(x)
            res = 0
            for c, idx in zip(coeffs, idxs):
                res += c * np.prod([polys[i](xi) for i, xi in zip(idx, x)])
            return res

        return lambda x: test_f(x, coeffs, polys, idxs, ip.gens)
    else:

        def test_f(x, coeffs, degrees, gen):
            x = gen.scale_back(x)
            res = 0
            for c, idx in zip(coeffs, idxs):
                res += c * np.prod(
                    [
                        np.polynomial.hermite.Hermite([0] * i + [1])(xi)
                        for i, xi in zip(idx, x)
                    ]
                )
            return res

        return lambda x: test_f(x, coeffs, idxs, ip.gens)


def test_tensorproduct_interpolation():

    for g in setup.generate_pointsets(10, 5):
        k = np.random.randint(low=1, high=7, size=g.d)

        print("Testing with d = {}, k = {}".format(g.d, k))
        g.print()
        k = {k: v for k, v in enumerate(k) if v > 0}

        ip = TensorProductBarycentricInterpolator(g, k, g.d)
        f = generate_test_function(ip, np.random.randint(low=1, high=10))
        ip.set_F(f)

        print("\t ... testing interpolation points")
        for x in it.product(*ip.nodes):
            x = np.array(x)
            y_f = f(x)
            y_i = ip(x)
            assert y_f.shape == y_i.shape
            assert np.isclose(
                y_i, y_f
            ).all(), f"Assertion failed with\n x = {x}\n f(x) = {y_f}\n ip(x) = {y_i}"

        print("\t ... testing random points")
        for n in range(100):
            r = np.random.randint(low=2, high=15)
            x = np.array([g.get_random() for _ in range(r)])
            x[0] = [nodes[0] for nodes in ip.nodes]
            x[-1] = [nodes[-1] for nodes in ip.nodes]
            y_f = np.array([f(xi) for xi in x])
            y_i = ip(x)
            assert y_f.shape == y_i.shape
            assert np.isclose(
                y_i, y_f
            ).all(), f"Assertion failed with\n x = {x}\n f(x) = {y_f}\n ip(x) = {y_i}"
