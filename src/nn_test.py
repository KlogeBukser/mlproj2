# nn_test.py
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from nn import *
from generate import *
from NNDebugger import *
from activation_funcs import *



# np.random.seed(1984)

def find_best_hyperparams(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd):
	# from adjust hyperparams
	"""
	min_learning_rate, max_learning_rate, min_lmbd, max_lmbd: int, defines range for hyperparameters in LOG SCALE
	"""

	find_params(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd, is_regressor=True)
	find_params(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd, is_regressor=False)



def find_params(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd, is_regressor):

	learning_rate_vals = np.logspace(min_learning_rate, max_learning_rate, max_learning_rate-min_learning_rate+1)
	lmbd_vals = np.logspace(min_lmbd, max_lmbd, max_lmbd-min_lmbd+1)

	data = {'learning rate' : {}, 'lambda' : {}, 'Score' : {}}
	df = pd.DataFrame(data)

	if is_regressor:

		xn,yn = simple_poly(1000)
		X_train, X_test, y_train, y_test = train_test_split(xn,yn, train_size=0.8, test_size=0.2)

		for learning_rate in learning_rate_vals:
			for lmbd in lmbd_vals:
				pred, score = my_regression(X_train, X_test, y_train, y_test,
					learning_rate=learning_rate, lmbd=lmbd, n_epochs=10, batch_size=25,
					activation="relu", activation_out="linear",is_debug=False, is_mse=True)
				df.loc[len(df.index)] = [np.log10(learning_rate),np.log10(lmbd),score]

		fig, ax = plt.subplots(figsize = (10, 10))
		sns.heatmap(df.pivot("learning rate", "lambda", "Score"), annot=True, ax=ax, cmap="viridis")
		ax.set_title("MSE scores (NN)")
		ax.set_ylabel("Learning rate: log$_{10}(\eta)$")
		ax.set_xlabel("Regularization parameter: log$_{10}(\lambda$)")
		plt.savefig("plots/hyperparams_regr.pdf")
		plt.show()

	else:

		X_train, X_test, y_train, y_test = cancer()

		for learning_rate in learning_rate_vals:
			for lmbd in lmbd_vals:
				score = my_classification(X_train, X_test, y_train, y_test,
					learning_rate=learning_rate, lmbd=lmbd,
					activation="sigmoid", activation_out="sigmoid",is_debug=False, is_show_cm=False)
				df.loc[len(df.index)] = [np.log10(learning_rate),np.log10(lmbd),score]

		fig, ax = plt.subplots(figsize = (10, 10))
		sns.heatmap(df.pivot("learning rate", "lambda", "Score"), annot=True, ax=ax, cmap="viridis")
		ax.set_title("Accuracy scores (NN)")
		ax.set_ylabel("Learning rate: log$_{10}(\eta)$")
		ax.set_xlabel("Regularization parameter: log$_{10}(\lambda$)")
		plt.savefig("plots/hyperparams_clf.pdf")
		plt.show()


# prepare data for training
def cancer():
	cancer_dataset = load_breast_cancer() # change to cancer data

	data = cancer_dataset.data
	target = cancer_dataset.target

	X_train, X_test, y_train, y_test = train_test_split(data,target, train_size=0.8, test_size=0.2)

	y_train = y_train.T.reshape(-1,1)
	y_test = y_test.T.reshape(-1,1)

	sc = StandardScaler()

	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)


	return X_train, X_test, y_train, y_test


def my_regression(X_train, X_test, y_train, y_test, 
	activation, activation_out, 
	learning_rate, lmbd, 
	n_epochs=10, batch_size=100, 
	is_debug=False, is_mse=False):
	nn = NNRegressor(X_train,y_train, 
		1, np.array([100]), 
		activation=activation,activation_out=activation_out, 
		learning_rate=learning_rate, lmbd=lmbd,
		n_epochs=n_epochs, batch_size=batch_size,
		is_debug=is_debug)
	nn.train()
	pred = nn.predict(X_test)
	if is_mse:
		return pred, nn.score(X_test, y_test)

	r2=nn.R2(X_test,y_test)
	print("myR2", r2)
	return pred, r2
	return pred, R2(X_test,y_test)

def sk_regression(X_train, X_test, y_train, y_test, activation):

	# result using sklearn
	sknn = MLPRegressor(activation=activation,random_state=1, max_iter=500)
	sknn.fit(X_train, y_train.ravel())
	sk_pred = sknn.predict(X_test)
	r2 = sknn.score(X_test, y_test)
	print("sklearn's R2", r2)

	return sk_pred, r2


def run_regression(activation_out, 
	coeffs=[3,2,1], noise_scale = 0.2,
	learning_rate=0.001, lmbd=0.001, 
	n_epochs=10, batch_size=100, 
	is_debug=False):

	x,yn,y = simple_poly(100, coeffs=coeffs, noise_scale = noise_scale, include_exact=True)
	X_train, X_test, y_train, y_test = train_test_split(x,yn, train_size=0.8, test_size=0.2)

	activations = ["sigmoid", "relu", "tanh", "leaky_relu"]

	for activation in activations:
		print(activation)
		pred, myr2 = my_regression(X_train, X_test, y_train, y_test, activation, activation_out, n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, lmbd=lmbd, is_debug=is_debug)
		if activation != "leaky_relu":
			if activation == "sigmoid":
				sk_pred, skr2 = sk_regression(X_train, X_test, y_train, y_test, "logistic")
			else:
				sk_pred, skr2 = sk_regression(X_train, X_test, y_train, y_test, activation)
	
		plt.plot(x,y, label="actual function", color='r')
		plt.scatter(X_test, y_test, label="y_test", s=5)
		plt.scatter(X_test, pred, label="my predition", s=8)
		if activation != "leaky_relu":
			plt.scatter(X_test, sk_pred, label="sklearn prediction", s=8)

		plt.title("Regression " + str(activation) + " R2 = " + str(myr2))
		plt.legend()
		plt.savefig("plots/regression-" + activation + ".pdf")
		plt.close()


def my_classification(X_train, X_test, y_train, y_test, 
	n_hidden_layers=1, n_nodes_in_layer=[100],
	n_catagories=1,
	learning_rate=0.001, lmbd=0.001,
	n_epochs=10, batch_size=100,
	activation="sigmoid", activation_out="sigmoid", is_debug=False, is_show_cm=True):
		
	clf = NNClassifier(X_train, y_train,
		n_hidden_layers, n_nodes_in_layer,
		n_catagories=n_catagories,
		activation=activation,activation_out=activation_out, 
		n_epochs=n_epochs, batch_size=batch_size,
		learning_rate=learning_rate, lmbd=lmbd,
		is_debug=is_debug)

	# nn.debugger.print_static()
	clf.train()
	result = clf.predict(X_test)
	acc = clf.score(X_test, y_test)
	print("My accuracy:", acc)

	if is_show_cm:
		cm = confusion_matrix(y_test, result)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm)
		disp.plot()
		plt.title("My Classification")
		plt.savefig("plots/my-clf.pdf")

	return acc


def sk_classification(X_train, X_test, y_train, y_test):
	clf = MLPClassifier()
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)
	counter = 0
	clf = MLPClassifier()
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)
	print("Sklearn's accuracy:", clf.score(X_test, y_test))
	cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
	disp.plot()
	plt.title("Sklearn's Classification")
	plt.savefig("plots/sk-clf.pdf")


def run_classification():

	X_train, X_test, y_train, y_test = cancer()

	my_classification(X_train, X_test, y_train, y_test, n_hidden_layers=2, n_nodes_in_layer=[100,100], n_epochs=10, batch_size=50, is_debug=False)

	sk_classification(X_train, X_test, y_train, y_test)


# ====================================Function Calls Below==============================================

# ignore stupid matplotlib warnings
warnings.filterwarnings("ignore" )

# regression


# run_regression("linear", coeffs=[3,0,1], noise_scale = 0.5, n_epochs=10, batch_size=10, learning_rate=0.001, lmbd=0.01, is_debug=False)

find_best_hyperparams(-6,6, -6,2)


# classification
# run_classification()





