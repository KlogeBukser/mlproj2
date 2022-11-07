from generate import gen_simple

from make_figure import *


n_datapoints = 100
x, y = gen_simple(n_datapoints)

sgd_figures(x,y,eta_method = 'basic', n_features = 3,n_iterations = 100)



