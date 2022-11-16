import numpy as np
from numpy.random import default_rng
from make_figure import *
from sklearn.datasets import load_breast_cancer

rng = default_rng(seed = 5473)


# Makes folder for holding plots if it doesn't already exist
plot_path = os.path.dirname(os.path.abspath(__file__)) + "/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


df = load_breast_cancer()

X = df.data
y = df.target[:,np.newaxis]

algos = ["SGD","ADA","RMS","ADAM"]
for algo in algos:
    hyper_matrix(X,y,min_rate = -6, max_rate = 6, min_lmb = -8, max_lmb = 2, n_batches = 5, n_epochs = 10, algo = algo)
