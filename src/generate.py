import numpy as np


def gen_simple(n):
    """Generates simple test data"""
    n = 100
    x = 2*np.random.rand(n,1)
    y = 4+3*x+np.random.randn(n,1)

    return x, y