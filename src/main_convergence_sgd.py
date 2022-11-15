from make_figure import convergence_plots
from generate import simple_poly
from misc import make_design_1D
import numpy as np

rng = np.random.default_rng(1)

# Generate data
n_datapoints = 100
x, y = simple_poly(n_datapoints)

# Make design matrix
n_features = 3
x,y = simple_poly(n_datapoints,coeffs = [3,0,1], noise_scale = 0.1)
X = make_design_1D(x,n_features)

# Makes the (6) figures SGDconvergence.pdf, SGD_0_mse.pdf, SGD_0.2_mse.pdf, SGD_0.4_mse.pdf, SGD_0.6_mse.pdf, SGD_0.8_mse.pdf
convergence_plots(X,y,algo = 'SGD', learning_rate = 0.1, batch_list = [1,2,5,10],bootstraps = 50)

# Makes the (6) figures ADAconvergence.pdf, ADA_0_mse.pdf, ADA_0.2_mse.pdf, ADA_0.4_mse.pdf, ADA_0.6_mse.pdf, ADA_0.8_mse.pdf
convergence_plots(X,y,algo = 'ADA', learning_rate = 0.5, batch_list = [1,2,10,50],bootstraps = 50)