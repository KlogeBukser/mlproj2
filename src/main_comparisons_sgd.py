from make_figure import comparison_plots, rates_plot
from generate import simple_poly
from misc import make_design_1D
import numpy as np

rng = np.random.default_rng(1)


# Make design matrix
n_features = 3
n_datapoints = 100
x,y = simple_poly(n_datapoints,coeffs = [3,2,1], noise_scale = 0.5)
X = make_design_1D(x,n_features)


# Comparison plots of several algorithms over learning rates and lambda values
comparison_plots(X,y,learning_rate_range = [0.01,1],lmbda_range = [-4,0],algos = ['SGD','ADA','RMS','ADAM'],n_epochs = 20,filename = "comparison1.pdf")
comparison_plots(X,y,learning_rate_range = [0.01,1],lmbda_range = [-4,0],algos = ['ADA','RMS','ADAM'],n_epochs = 20,filename = "comparison2.pdf")
comparison_plots(X,y,learning_rate_range = [0.01,1],lmbda_range = [-4,-2],algos = ['ADA','RMS','ADAM'],n_epochs = 20,filename = "comparison3.pdf")
comparison_plots(X,y,learning_rate_range = [0.01,0.15],lmbda_range = [-4,0],algos = ['SGD'],n_epochs = 20,filename = "comparison4.pdf")


# Plot over selected ranges of learning rates for all four algorithms with 1 and 2 minibatches
rates_plot(X, y, learning_rates = np.linspace(0.01, 0.17, 100), algo = 'SGD',n_batches=1)
rates_plot(X, y, learning_rates = np.linspace(0.01, 0.3, 100), algo = 'RMS',n_batches=1)
rates_plot(X, y, learning_rates = np.linspace(0.4, 5, 100), algo = 'ADA',n_batches=1)
rates_plot(X, y, learning_rates = np.linspace(0.01, 5, 100), algo = 'ADAM',n_batches=1)

rates_plot(X, y, learning_rates = np.linspace(0.01, 0.17, 100), algo = 'SGD',n_batches=2)
rates_plot(X, y, learning_rates = np.linspace(0.01, 0.3, 100), algo = 'RMS',n_batches=2)
rates_plot(X, y, learning_rates = np.linspace(0.4, 5, 100), algo = 'ADA',n_batches=2)
rates_plot(X, y, learning_rates = np.linspace(0.01, 5, 100), algo = 'ADAM',n_batches=2)
