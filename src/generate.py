import numpy as np


def gen_simple(n):
    """Generates simple test data"""
    x = 2*np.random.rand(n,1)
    y = 4+3*x + 2*np.square(x)+np.random.randn(n,1)

    return x, y