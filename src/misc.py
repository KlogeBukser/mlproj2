import numpy as np
from poly_funcs import get_1D_pols
from learning_rates import *

def make_design_1D(x,n_features):
    """ (From project 1 with some adjustments)
    """
    # Uses the features to turn set of tuple values (x,y) into design matrix
    n = x.shape[0]
    design = np.ones((n, n_features))
    design_functions = get_1D_pols(n_features)
    for i in range(n):
        for j in range(n_features):
            design[i,j] = design_functions[j](x[i])

    return design

def make_adaptive_learner(eta_method = 'basic', n_features = 3, learning_rate = 0.01):
    if eta_method == 'ada':
        return ADA(n_features,learning_rate)
    if eta_method == 'rms':
        return RMSProp(n_features,learning_rate)
    if eta_method == 'adam':
        return RMSProp(n_features,learning_rate)

    return Basic_learning_rate(learning_rate)

def R2(z_test,z_pred):
    """computes the mean squared error for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: R squared score

    """

    return 1 - np.mean((z_test - z_pred)**2)/np.mean((z_test - np.mean(z_test))**2)


# some of these are from lecture notes with modification
def MSE(z_test, z_pred):
    """computes the mean squared error for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: Mean squared error

    """

    return np.mean((z_test - z_pred)**2)

def MSE_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def cal_bias(z_test, z_pred):
    """computes the bias for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: bias

    """
    return np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )

def cal_variance(z_pred):
    """computes the variance for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: variance

    """
    return np.mean( np.var(z_pred, axis=1, keepdims=True) )