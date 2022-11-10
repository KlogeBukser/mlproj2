from generate import gen_simple
from hyperparams import Hyperparams
from make_figure import make_sgd_compare_plot

n_datapoints = 300
x, y = gen_simple(n_datapoints)

n_features = 3
n_predictions = 200
n_iterations = 200


params = Hyperparams()

make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions)

params.change_limits('learning_rates',0.15,upper = True)

make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions)

params.rm_algo('ada')

make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions)

params.change_limits('learning_rates',0.05,upper = False)

make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions)

params.change_limits('lmbdas',-2,upper = True)

make_sgd_compare_plot(x, y, params, n_features, n_iterations, n_predictions)