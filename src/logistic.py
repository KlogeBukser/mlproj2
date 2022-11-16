import numpy as np
from numpy.random import default_rng
from make_figure import *
from sklearn.datasets import load_breast_cancer
from generate import simple_poly

rng = default_rng(seed = 5473)


# Makes folder for holding plots if it doesn't already exist
plot_path = os.path.dirname(os.path.abspath(__file__)) + "/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


# Regression
# Generate data
n_datapoints = 100
n_features = 3
x,y = simple_poly(n_datapoints,coeffs = [3,0,1], noise_scale = 0.5)
X = make_design_1D(x,n_features)


algos = ["SGD"]
for algo in algos:
    hyper_matrix(X,y,min_rate = -5, max_rate = -1, min_lmb = -8, max_lmb = 1, n_batches = 10, n_epochs = 200, algo = algo,logistic = False, rate_points=4)    



# Classification
df = load_breast_cancer()

X = df.data
y = df.target[:,np.newaxis]

algos = ["SGD","ADA","RMS","ADAM"]
for algo in algos:
    hyper_matrix(X,y,min_rate = -6, max_rate = 6, min_lmb = -8, max_lmb = 2, n_batches = 5, n_epochs = 10, algo = algo,logistic = True, rate_points=1)
