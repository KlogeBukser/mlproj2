from make_figure import comparison_plots, rates_plot
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


# Comparison plots of several algorithms over learning rates and lambda values
comparison_plots(X,y,learning_rate_range = [0.01,1],lmbda_range = [-4,0],algos = ['SGD','ADA','RMS','ADAM'],filename = "comparison1.png")
comparison_plots(X,y,learning_rate_range = [0.01,1],lmbda_range = [-4,0],algos = ['ADA','RMS','ADAM'],filename = "comparison2.png")
comparison_plots(X,y,learning_rate_range = [0.01,1],lmbda_range = [-4,-2],algos = ['ADA','RMS','ADAM'],filename = "comparison3.png")
comparison_plots(X,y,learning_rate_range = [0.01,0.15],lmbda_range = [-4,0],algos = ['SGD'],filename = "comparison4.png")


# Plot over selected ranges of learning rates for all four algorithms with 1 and 2 minibatches
rates_plot(X, y, learning_rates = np.linspace(0.01, 0.17, 200), algo = 'SGD',n_batches=1)
rates_plot(X, y, learning_rates = np.linspace(0.01, 0.3, 200), algo = 'RMS',n_batches=1)
rates_plot(X, y, learning_rates = np.linspace(0.4, 5, 200), algo = 'ADA',n_batches=1)
rates_plot(X, y, learning_rates = np.linspace(0.01, 5, 200), algo = 'ADAM',n_batches=1)

rates_plot(X, y, learning_rates = np.linspace(0.01, 0.17, 200), algo = 'SGD',n_batches=2)
rates_plot(X, y, learning_rates = np.linspace(0.01, 0.3, 200), algo = 'RMS',n_batches=2)
rates_plot(X, y, learning_rates = np.linspace(0.4, 5, 200), algo = 'ADA',n_batches=2)
rates_plot(X, y, learning_rates = np.linspace(0.01, 5, 200), algo = 'ADAM',n_batches=2)
