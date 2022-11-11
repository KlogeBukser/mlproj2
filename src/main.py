from generate import gen_simple
import numpy as np
from numpy.random import default_rng
from make_figure import *
from hyperparams import Hyperparams

rng = default_rng(seed = 5473)

plot_path = os.path.dirname(os.path.abspath(__file__)) + "/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# Making simple plots
def simple_plots():
    # Collecting and preparing data + setting initial parameters
    n_datapoints = 50
    x, y = gen_simple(n_datapoints)
    n_features = 3
    n_epochs = 200
    lmbda = 0
    learning_rate = 0.1

    algo = 'adam'
    momentums = np.arange(0,0.6,0.1)
    smooth = False
    n_batches = 1
    momentum_plot(x,y,learning_rate,n_epochs,n_features,lmbda,n_batches,momentums, algo,smooth)


    n_batches_list = np.arange(1,11,3)
    momentum = 0
    batches_plot(x,y,learning_rate,n_epochs,n_features,lmbda,n_batches_list,momentum , algo, smooth)


# Seaborn plots
def comparison_plots():
    n_datapoints = 300
    x, y = gen_simple(n_datapoints)

    n_features = 3
    n_predictions = 200
    n_iterations = 200


    params = Hyperparams()

    make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions, 'comp1.png')


    params.change_limits('learning_rates',0.15,upper = True)

    make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions, 'comp2.png')

    params.rm_algo('ada')

    make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions, 'comp3.png')

    params.change_limits('learning_rates',0.05,upper = False)

    make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions, 'comp4.png')

    params.change_limits('lmbdas',-2,upper = True)

    make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions, 'comp5.png')

simple_plots()
#comparison_plots()