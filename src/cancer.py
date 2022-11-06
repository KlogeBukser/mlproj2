# cancer.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import jax.numpy as jnp
import seaborn as sns


def basic_nn_pred(eta, lmbd):
	nn = NeuralNetwork(1,1,1,1, eta, lmbd)
	nn.train()
	pred = nn.predict()

	# score = cal_accuracy(pred, test)

def find_best_hyperparams(min_eta, max_eta, min_lmbd, max_lmbd):
	# from adjust hyperparams
	"""blah
	min_eta, max_eta, min_lmbd, max_lmbd: int, defines range for hyperparameters in LOG SCALE
	"""

	dnn_container = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

	eta_vals = np.logspace(min_eta, max_eta, 8)
	lmbd_vals = np.logspace(min_lmbd, max_lmbd, 8)

	for i, eta in enumerate(eta_vals):
		continue


def visualiser():
	# seeborn heat map learning_rate vs lambda
	pass



# prepare data for training
cancer_dataset = load_breast_cancer() # change to cancer data

data = cancer_dataset.data
target = cancer_dataset.target
labels = cancer_dataset.feature_names


X_train, X_test, y_train, y_test = train_test_split(data,target, train_size=0.8, test_size=0.2)


# flatten
# jnp.ravel(images)

# result using my_nn

# result using sklearn