import numpy as np

def simple_poly(n_points, coeffs = [3,2,1], noise_scale = 0.5, include_exact = False):
    """ Generates data for one variable polynomial """

    rng = np.random.default_rng()
    x = np.linspace([0],[2],n_points)
    n_vars = len(coeffs)
    noise = noise_scale*rng.normal(0,1,(n_points,1))
    y_exact = sum([coeffs[i]*x**i for i in range(n_vars)])
    y = y_exact + noise

    if include_exact:
        return x, y, y_exact
    return x, y
