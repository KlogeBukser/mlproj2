# nn_test.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from nn import *
from generate import gen_simple
import matplotlib.pyplot as plt
from NNDebugger import *

from sklearn.neural_network import MLPRegressor


# np.random.seed(437)


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
def cancer():
	cancer_dataset = load_breast_cancer() # change to cancer data

	data = cancer_dataset.data
	target = cancer_dataset.target
	labels = cancer_dataset.feature_names

	X_train, X_test, y_train, y_test = train_test_split(data,target, train_size=0.8, test_size=0.2)


# flatten
# jnp.ravel(images)

# result using my_nn
# regression 
# basic_nn_pred(0.001,0)

# classification

x,y = gen_simple(10000)
learning_rate = 0.001
lmbd = 0.0001
X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2)
# print("y_train is",y_train.shape)

nn = NNRegressor(X_train,y_train, 3, np.array([10,10,10]), learning_rate=learning_rate, lmbd=lmbd, is_debug=True)
nn.debugger.print_static()
nn.train()
pred = nn.predict(X_test)


plt.scatter(X_test, y_test, label="true values")
plt.scatter(X_test, pred, label="my regressor")
# plt.show()

# result using sklearn
# regression

regr = MLPRegressor(random_state=1, max_iter=500)
regr.fit(X_train, y_train)
re_pred = regr.predict(X_test)
print(X_test.shape, y_test.shape)
# plt.scatter(X_test, y_test)
plt.scatter(X_test, re_pred, label="sklearn regressor")

plt.legend()
plt.show()


# classification