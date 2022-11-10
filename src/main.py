from generate import gen_simple

from make_figure import *

eta_method = 'adam'
n_datapoints = 100
x, y = gen_simple(n_datapoints)

sgd_figures(x,y,eta_method, n_features = 3,n_iterations = 100)

n_features = 3
n_predictions = 200
n_iterations = 100

make_sgd_pairplot(eta_method, x, y, n_features, n_iterations, n_predictions)



