# nn_test.py
from sklearn.datasets import load_breast_cancer
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



# np.random.seed(437)


def find_best_hyperparams(min_eta, max_eta, min_lmbd, max_lmbd):
	# from adjust hyperparams
	"""blah
	min_eta, max_eta, min_lmbd, max_lmbd: int, defines range for hyperparameters in LOG SCALE
	"""

	nn_container = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

	eta_vals = np.logspace(min_eta, max_eta, 8)
	lmbd_vals = np.logspace(min_lmbd, max_lmbd, 8)

	for i, eta in enumerate(eta_vals):
		for j, lmbd_vals in enumerate(lmbd_vals):
			continue



def visualiser():
	# seeborn heat map learning_rate vs lambda
	learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
	lmbds = [0.00001, 0.0001, 0.001, 0.01, 0.1]

	for learning_rate in learning_rates:
		for lmbds in lmbds:
			continue


# prepare data for training
def cancer():
	cancer_dataset = load_breast_cancer() # change to cancer data

	data = cancer_dataset.data
	target = cancer_dataset.target
	labels = cancer_dataset.feature_names

	X_train, X_test, y_train, y_test = train_test_split(data,target, train_size=0.8, test_size=0.2)

	return X_train, X_test, y_train, y_test

# result using my_nn
# regression 
# basic_nn_pred(0.001,0)

def my_regression(X_train, X_test, y_train, y_test, activation, activation_out, learning_rate, lmbd, is_debug=False):
	nn = NNRegressor(X_train,y_train, 
		1, np.array([100]), 
		activation=activation,activation_out=activation_out, 
		learning_rate=learning_rate, lmbd=lmbd,
		n_epochs=100, batch_size=100,
		is_debug=is_debug)
	nn.debugger.print_static()
	nn.train()
	pred = nn.predict(X_test)
	r2 = nn.R2(X_test,y_test)
	print("myR2",r2)

	return pred, r2

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
		plt.savefig("plots/regression-" + activation + ".png")
		plt.close()


def run_classification():

	X_train, X_test, y_train, y_test = cancer()

	# my_classification(X_train, X_test, y_train, y_test)
	# sk_classification(X_train, X_test, y_train, y_test)

	cancer_results = []
	for i in range(1):
		sk_classification(X_train, X_test, y_train, y_test)
	# print(cancer_results)

# ====================================Function Calls Starting Here==============================================

def my_classification(X_train, X_test, y_train, y_test, 
	n_catagories=1,
	learning_rate=0.001, lmbd=0.001,
	activation="sigmoid", activation_out="sigmoid", is_debug=True):
		
	y_train = y_train.T.reshape(-1,1)
	print(y_train.shape)
	nn = NNClassifier(X_train, y_train,
		1, np.array([100]), 
		n_catagories=n_catagories,
		activation=activation,activation_out=activation_out, 
		n_epochs=100, batch_size=10,
		learning_rate=learning_rate, lmbd=lmbd,
		is_debug=is_debug)

	nn.debugger.print_static()
	nn.train()
	result = nn.predict(X_test)


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
	  #print(predictions)
	  #print(y_test)
	cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
	disp.plot()
	plt.show()

	

# ignore stupid matplotlib warnings
warnings.filterwarnings("ignore" )

# regression
activations = ["sigmoid", "relu", "tanh"] #, "leaky_relu"]
# run_regression("linear")


# classification
run_classification()

# MAX_STEP = 10
# X_train, X_test, y_train, y_test = cancer()
# y_train = y_train.T.reshape(-1,1) # super important!!!!
# myclf = NNClassifier(X_train, y_train,
# 		1, np.array([100]), 
# 		n_catagories=1,
# 		activation="sigmoid",activation_out="sigmoid", 
# 		n_epochs=10, batch_size=10,
# 		learning_rate=0.001, lmbd=0.001,
# 		is_debug=True)
# # myclf.train()
# for i in range(MAX_STEP):
# 	myclf.prep()
# 	myclf.feed_forward()
# 	myclf.back_propagate()	




