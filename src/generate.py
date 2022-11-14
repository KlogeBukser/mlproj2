import numpy as np

def simple_poly(n_points, coeffs = [3,4], noise_scale = 0.5):
    """ Generates data for one variable polynomial """

    rng = np.random.default_rng()
    x = np.linspace([0],[2],n_points)
    n_vars = len(coeffs)
    noise = noise_scale*rng.normal(0,1,(n_points,1))
    y = sum([coeffs[i]*x**i for i in range(n_vars)]) + noise

    return x,y

def gen_simple(n_points, is_noiseless=False):
    rng = np.random.default_rng()
    """Generates simple test data"""
    xn = np.linspace([0],[2],n_points)
    yn = 4*xn+3 + rng.normal(0,1,(n_points,1))*0.5

    x = np.arange(0,2,0.01)
    y = 4*x+3

    if is_noiseless:
        return xn,yn,x,y
    return xn, yn



def gen_simple2(n_points, is_noiseless=False):
    rng = np.random.default_rng()
    """Generates simple test data"""
    xn = np.linspace([0],[2],n_points)
    yn = xn**2 + 3 + rng.normal(0,1,(n_points,1))*0.2

    x = np.arange(0,2,0.01)
    y = x**2 + 3

    if is_noiseless:
        return xn,yn,x,y
    return xn, yn