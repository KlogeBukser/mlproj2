# nn_test.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import jax.numpy as jnp
import seaborn as sns
from nn import *
from generate import gen_simple



def basic_nn_pred(learning_rate, lmbd):
	x,y = gen_simple(1000)
	# print(x)
	# X_inputs, 
	# Y_inputs,  
	# n_hidden_layers, 
	# n_nodes, 
	# n_catagories,
	# acti_func_out=sigmoid,
	# n_epochs=10, 
	# batch_size=100, 
	# learning_rate=0.01, 
	# lmbd=0.0,


	X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2)
	# print("y_train is",y_train.shape)

	nn = NNRegressor(X_train,y_train, 2, np.array([50,50]), learning_rate, lmbd)
	nn.train()
	pred = nn.predict(X_test)
	# print(X_test)
	# print(y_test)
	print(pred)

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
# regression 
basic_nn_pred(0.0001, 0.0)

# classification


# result using sklearn
# regression

# classification