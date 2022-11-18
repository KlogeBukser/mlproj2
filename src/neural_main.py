import warnings
import numpy as np
from nn_test import *

np.random.seed(1984)
# ignore matplotlib warnings
warnings.filterwarnings("ignore" )

# regression
print("================================= Regression =================================")
# myr2, skr2 = run_regression("linear", coeffs=[3,2,1], noise_scale = 0.5, n_epochs=20, batch_size=20, learning_rate=0.001, lmbd=0.001, is_debug=False)
# cmp(run_regression, "sigmoid", "nn-regr-cmp-sig.pdf", "Performance comparison for neural networks sigmoid (Regression)", "R2 score")
# cmp(run_regression, "relu", "nn-regr-cmp-relu.pdf", "Performance comparison for neural networks relu", "R2 score")
cmp(run_regression, "tanh", "nn-regr-cmp-tanh.pdf", "Performance comparison for neural networks tanh", "R2 score")

# classification
print("================================= Classification =================================")
run_classification()

# print("================================= Hyperparameters =================================")
# find_best_hyperparams(-10,6, -6,2)


cmp(run_classification, "sigmoid", "nn-clf-cmp-sig.pdf", "Performance comparison for neural networks (Classification)", "Accuracy score")
