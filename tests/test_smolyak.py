import numpy as np
import setup

from smolyax.interpolation import SmolyakBarycentricInterpolator


def test_interpolation():
    print("\nTesting vector-valued Smolyak interpolation ...")

    for node_gen in setup.generate_nodes(n=5, dmin=1, dmax=4):

        k = sorted(np.random.uniform(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        d_out = np.random.randint(low=1, high=4)
        t = np.random.uniform(low=1, high=8)
        print(f"... with k = {k}, t = {t}, d_out = {d_out},", node_gen)

        f = setup.TestPolynomial(node_gen=node_gen, k=k, t=t, d_out=d_out)

        ip = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=d_out, batchsize=1, f=f)

        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            assert np.allclose(
                f(x), ip(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"


def test_quadrature():
    print("\nTesting vector-valued Smolyak quadrature ...")

    for node_gen in setup.generate_nodes(n=5, dmin=1, dmax=4):

        k = sorted(np.random.uniform(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        d_out = np.random.randint(low=1, high=4)
        t = np.random.uniform(low=1, high=8)
        print(f"... with k = {k}, t = {t}, d_out = {d_out},", node_gen)

        f = setup.TestPolynomial(node_gen=node_gen, k=k, t=t, d_out=d_out)

        ip = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=d_out, batchsize=1, f=f)

        Q_ip = ip.integral()
        assert Q_ip.shape == (d_out,)

        Q_f = f.integral()
        print(f"\tIntegral of f = {Q_f}\n\tQ[f] = {Q_ip}")
        assert np.allclose(Q_ip, Q_f), f"Assertion failed with\n Q[f] = {Q_ip}\n Integral of f = {Q_f}"


def test_gradient():
    print("\nTesting gradients of the vector-valued Smolyak interpolant ...")

    for node_gen in setup.generate_nodes(n=5, dmin=1, dmax=4):

        k = sorted(np.random.uniform(low=1, high=10, size=node_gen.dim))
        k /= k[0]
        d_out = np.random.randint(low=1, high=4)
        t = np.random.uniform(low=1, high=8)
        print(f"... with k = {k}, t = {t}, d_out = {d_out},", node_gen)

        f = setup.TestPolynomial(node_gen=node_gen, k=k, t=t, d_out=d_out)

        ip = SmolyakBarycentricInterpolator(node_gen=node_gen, k=k, t=t, d_out=d_out, batchsize=1, f=f)

        for n in range(5):
            x = node_gen.get_random(n=np.random.randint(low=1, high=5))
            grad_f = f.gradient(x)
            grad_ip = ip.gradient(x)
            assert (
                grad_f.shape == grad_ip.shape
            ), f"Assertion failed with\n x = {x}\n grad f(x) = {grad_f}\n grad ip(x) = {grad_ip} @ n = {n}"
            assert np.allclose(
                grad_f, grad_ip
            ), f"Assertion failed with\n x = {x}\n grad f(x) = {grad_f}\n grad ip(x) = {grad_ip} @ n = {n}"
