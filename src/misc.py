import numpy as np
from poly_funcs import get_1D_pols

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