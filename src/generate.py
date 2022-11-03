import numpy as np


def gen_simple(n_variables,n_points):
    rng = np.random.default_rng()
    """Generates simple test data"""
    x = 2*np.random.rand(n_points,1)
    y = np.random.randn(n_points,1)
    for i in range(n_variables):
        y += rng.integers(0,5)*x**i

    return x, y