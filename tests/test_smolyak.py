import setup

from jax_smolyak.smolyak import *


def test_smolyak_scalar():
    print("\nTesting scalar-valued Smolyak operator (numpy) ...")

    for g in setup.generate_pointsets(n=10, dmin=1, dmax=4):

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        t = np.random.randint(low=1, high=4)
        print(f"... with k = {k}, t = {t},", g)

        ip = SmolyakBarycentricInterpolator(g, k, t)
        ff = setup.generate_test_function_smolyak(g=g, k=k, t=t, d_out=1)
        f = lambda x: np.squeeze(ff(x))
        ip.set_F(f)

        for n in range(5):
            x = g.get_random(n=np.random.randint(low=0, high=5))
            assert np.allclose(
                ip(x), f(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}"


def test_smolyak_vector():
    print("\nTesting vector-valued Smolyak operator (numpy) ...")

    for g in setup.generate_pointsets(n=10, dmin=1, dmax=4):

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        d_out = np.random.randint(low=1, high=5)
        t = sorted(np.random.randint(low=1, high=4, size=d_out), reverse=True)
        print(f"... with k = {k}, t = {np.array(t).tolist()},", g)

        ip = MultivariateSmolyakBarycentricInterpolator(g=g, k=k, t=t)
        f = setup.generate_test_function_smolyak(g=g, k=k, t=t, d_out=d_out)
        ip.set_F(f=f)

        for n in range(5):
            x = g.get_random(n=np.random.randint(low=0, high=5))
            print(f"\t\t ip(x) = {ip(x)}, f(x) = {f(x)}")
            assert np.allclose(
                ip(x), f(x)
            ), f"Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}"
