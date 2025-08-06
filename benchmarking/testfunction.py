from typing import List

import numpy as np

BASE_R = 2.0
BASE_THETA = 0.1
NOISE = 0.3


def build_test_function(d_in: int, d_out: int):
    rng = np.random.default_rng(0)
    noise_r = rng.uniform(-BASE_R * NOISE, BASE_R * NOISE, d_out)
    noise_theta = rng.uniform(-BASE_THETA * NOISE, BASE_THETA * NOISE, d_out)

    r = BASE_R + noise_r
    theta = BASE_THETA + noise_theta
    r[0] = BASE_R
    theta[0] = BASE_THETA

    j_mat = (np.arange(d_in) + 2) ** (-r[:, None])

    def f(x, tid: int | None = None):
        x_arr = np.asarray(x)
        dot = x_arr @ j_mat.T
        return 1.0 / (1.0 + theta * dot)

    return f, BASE_R, BASE_THETA


