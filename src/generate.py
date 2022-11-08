import numpy as np


def gen_simple(n_points):
    rng = np.random.default_rng()
    """Generates simple test data"""
    x = np.linspace([0],[2],n_points)
    y = 4 + 3*x  + rng.normal(0,1,(n_points,1))*0.2

    return x, y