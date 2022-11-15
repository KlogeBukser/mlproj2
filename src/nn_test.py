# nn_test.py
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from nn import *
from generate import *
from NNDebugger import *
from activation_funcs import *



np.random.seed(1984)

def find_best_hyperparams(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd):
	# from adjust hyperparams
	"""blah
	min_learning_rate, max_learning_rate, min_lmbd, max_lmbd: int, defines range for hyperparameters in LOG SCALE
	"""

	# find_params(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd, is_regressor=True)
	find_params(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd, is_regressor=False)



def find_params(min_learning_rate, max_learning_rate, min_lmbd, max_lmbd, is_regressor):

	if is_regressor:

		learning_rate_vals = np.logspace(min_learning_rate, max_learning_rate, max_learning_rate-min_learning_rate+1)
		lmbd_vals = np.logspace(min_lmbd, max_lmbd, max_lmbd-min_lmbd+1)

		mse_scores = np.zeros((len(learning_rate_vals), len(lmbd_vals)))

		xn,yn = gen_simple2(2000)
		X_train, X_test, y_train, y_test = train_test_split(xn,yn, train_size=0.8, test_size=0.2)

		for i, learning_rate in enumerate(learning_rate_vals):
			for j, lmbd in enumerate(lmbd_vals):
				pred, mse_scores[i][j] = my_regression(X_train, X_test, y_train, y_test,
					learning_rate=learning_rate, lmbd=lmbd,
					activation="relu", activation_out="linear",is_debug=False, is_mse=True)

		fig, ax = plt.subplots(figsize = (10, 10))
		sns.heatmap(mse_scores, annot=True, ax=ax, cmap="viridis")
		ax.set_title("MSE scores")
		ax.set_ylabel("$\eta$")
		ax.set_xlabel("$\lambda$")
		plt.savefig("hyperparams_regr.pdf")
		plt.show()

	else:
		learning_rate_vals = np.logspace(min_learning_rate, max_learning_rate, max_learning_rate-min_learning_rate+1)
		lmbd_vals = np.logspace(min_lmbd, max_lmbd, max_lmbd-min_lmbd+1)

		X_train, X_test, y_train, y_test = cancer()
		accuracy_scores = np.zeros((len(learning_rate_vals), len(lmbd_vals)))

		for i, learning_rate in enumerate(learning_rate_vals):
			for j, lmbd in enumerate(lmbd_vals):
				accuracy_scores[i][j] = my_classification(X_train, X_test, y_train, y_test,
					learning_rate=learning_rate, lmbd=lmbd,
					activation="sigmoid", activation_out="sigmoid",is_debug=False, is_show_cm=False)

		fig, ax = plt.subplots(figsize = (10, 10))
		sns.heatmap(accuracy_scores, annot=True, ax=ax, cmap="viridis")
		ax.set_title("Accuracy scores")
		ax.set_ylabel("$\eta$")
		ax.set_xlabel("$\lambda$")
		plt.savefig("hyperparams_clf.pdf")
		plt.show()


# prepare data for training
def cancer():
	cancer_dataset = load_breast_cancer() # change to cancer data

	data = cancer_dataset.data
	target = cancer_dataset.target
	labels = cancer_dataset.feature_names

	# print("here", [(labels[i],np.median(data[i])) for i in range(len(labels))])
	X_train, X_test, y_train, y_test = train_test_split(data,target, train_size=0.8, test_size=0.2)
	# y_train_onehot, y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

	y_train = y_train.T.reshape(-1,1)
	y_test = y_test.T.reshape(-1,1)

	sc = StandardScaler()

	# print(X_test[0:3][:10])

	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)


	return X_train, X_test, y_train, y_test

# result using my_nn
# regression 
# basic_nn_pred(0.001,0)

def my_regression(X_train, X_test, y_train, y_test, activation, activation_out, learning_rate, lmbd, is_debug=False, is_mse=False):
	nn = NNRegressor(X_train,y_train, 
		1, np.array([100]), 
		activation=activation,activation_out=activation_out, 
		learning_rate=learning_rate, lmbd=lmbd,
		n_epochs=100, batch_size=100,
		is_debug=is_debug)
	nn.debugger.print_static()
	nn.train()
	pred = nn.predict(X_test)
	if is_mse:
		return pred, nn.score(X_test, y_test)

	return pred, R2(X_test,y_test)

def sk_regression(X_train, X_test, y_train, y_test, activation):

	# result using sklearn
	sknn = MLPRegressor(activation=activation,random_state=1, max_iter=500)
	sknn.fit(X_train, y_train.ravel())
	sk_pred = sknn.predict(X_test)
	r2 = sknn.score(X_test, y_test)
	print("sklearn's R2", r2)

	return sk_pred, r2


def run_regression(activation_out, learning_rate=0.001, lmbd=0.001):
	xn,yn,x,y = gen_simple2(2000, True)
	X_train, X_test, y_train, y_test = train_test_split(xn,yn, train_size=0.8, test_size=0.2)


	for activation in activations:
		print(activation)
		pred, myr2 = my_regression(X_train, X_test, y_train, y_test, "tanh", activation_out, learning_rate=learning_rate, lmbd=lmbd)
		if activation == "sigmoid":
			sk_pred, skr2 = sk_regression(X_train, X_test, y_train, y_test, "logistic")
		else:
			sk_pred, skr2 = sk_regression(X_train, X_test, y_train, y_test, activation)
	
		plt.plot(x,y, label="actual function", color='r')
		plt.scatter(X_test, y_test, label="y_test", s=5)
		plt.scatter(X_test, pred, label="my predition", s=8)
		plt.scatter(X_test, sk_pred, label="sklearn prediction", s=8)

		plt.title("Regression " + str(activation) + " R2 = " + str(myr2))
		plt.legend()
		# plt.show()
		plt.savefig("plots/regression-" + activation + ".pdf")
		plt.close()


def my_classification(X_train, X_test, y_train, y_test, 
	n_catagories=1,
	learning_rate=0.001, lmbd=0.001,
	n_epochs=10, batch_size=100,
	activation="sigmoid", activation_out="sigmoid", is_debug=False, is_show_cm=True):
		

	# print(y_train.shape)
	clf = NNClassifier(X_train, y_train,
		1, np.array([100]), 
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
	#print(predictions)
	#print(y_test)
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

	my_classification(X_train, X_test, y_train, y_test, n_epochs=10, batch_size=100, is_debug=False)

	sk_classification(X_train, X_test, y_train, y_test)


# ====================================Function Calls Below==============================================

# ignore stupid matplotlib warnings
warnings.filterwarnings("ignore" )

# regression
# activations = ["sigmoid", "relu", "tanh"] #, "leaky_relu"]
# run_regression("linear")
find_best_hyperparams(-9,2, -9,2)


# classification
run_classification()





