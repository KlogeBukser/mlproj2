from generate import gen_simple

from make_figure import *

eta_method = 'adam'
n_datapoints = 300
x, y = gen_simple(n_datapoints)

n_features = 3
n_predictions = 200
n_iterations = 200

make_sgd_compare_plot(x, y, n_features, n_iterations, n_predictions)

make_epoch_plot(x, y, n_iterations,n_features,lmbda = 0,eta_method = 'adam',n_batches = 1,learning_rate = 0.1)




