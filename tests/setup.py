import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.hermite import hermval

from jax_smolyak import points, indices


def generate_pointsets(*, n, dmin, dmax):

    sets = []
    for _ in range(n):

        d = np.random.randint(low=dmin, high=dmax + 1)

        m = np.random.randn(d)
        a = np.random.rand(d)
        sets.append(points.GaussHermiteMulti(m, a))

        domain = np.zeros((d, 2))
        domain[:, 1] = np.sort(np.random.rand(d) * 10)
        domain[:, 0] = -domain[:, 1]
        sets.append(points.LejaMulti(domains=domain))

    return sets


def evaluate_univariate_polynomial(g, nu, x):
    x = g.scale_back(x)
    if isinstance(g, points.LejaMulti):
        return legval(x, [0] * nu + [1])
    else:
        return hermval(x, [0] * nu + [1])


def evaluate_multivariate_polynomial(g, nu, x):
    if x.ndim <= 1:
        return np.prod(
            [
                evaluate_univariate_polynomial(gi, nui, xi)
                for gi, nui, xi in zip(g, nu, x)
            ]
        )
    elif x.ndim == 2:
        return np.prod(
            np.array(
                [
                    evaluate_univariate_polynomial(gi, nui, xi)
                    for gi, nui, xi in zip(g, nu, x.T)
                ]
            ),
            axis=0,
        ).T
    else:
        raise


def generate_test_function_tensorproduct(*, g, nu):
    if np.isscalar(nu):
        return lambda x: evaluate_univariate_polynomial(g, nu, x)
    return lambda x: evaluate_multivariate_polynomial(g, nu, x)


def generate_test_function_smolyak(*, g, k, l, d_out):
    if np.isscalar(l):
        l = [l] * d_out
    assert len(l) == d_out

    selected_idxs = []
    for li in l:
        idxs = indices.indexset_sparse(lambda j: k[j], li, cutoff=len(k))
        j = np.random.randint(len(idxs))
        selected_idxs.append(indices.sparse_index_to_dense(idxs[j], cutoff=len(k)))
    print("\t Test polynomials with degrees", selected_idxs)
    return lambda x: np.array(
        [evaluate_multivariate_polynomial(g, nu, x) for nu in selected_idxs]
    ).T
