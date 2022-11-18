import warnings
from nn_test import *

# ignore stupid matplotlib warnings
warnings.filterwarnings("ignore" )

# regression
run_regression("linear", coeffs=[3,2,1], noise_scale = 0.5, n_epochs=20, batch_size=50, learning_rate=0.001, lmbd=0.0, is_debug=True)
# find_best_hyperparams(-10,6, -6,2)


# classification
# run_classification()