from generate import gen_simple

from make_figure import *


n_datapoints = 100
x, y = gen_simple(n_datapoints)

sgd_figures(x,y,eta_method = 'basic', n_features = 3,n_iterations = 100)


n_predictions = 200

df = make_dataframe_sgd(x, y, eta_method = 'basic', n_features = 3,n_iterations = 100, n_predictions = n_predictions)

make_sgd_pairplot(df,method = 'basic')

df = make_dataframe_sgd(x, y, eta_method = 'ada', n_features = 3,n_iterations = 100, n_predictions = n_predictions)

make_sgd_pairplot(df,method = 'ada')

df = make_dataframe_sgd(x, y, eta_method = 'rms', n_features = 3,n_iterations = 100, n_predictions = n_predictions)

make_sgd_pairplot(df,method = 'rms')

df = make_dataframe_sgd(x, y, eta_method = 'adam', n_features = 3,n_iterations = 100, n_predictions = n_predictions)

make_sgd_pairplot(df,method = 'adam')



