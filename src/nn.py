import numpy as np
from activation_funcs import ActivationFunction

from misc import MSE, MSE_prime, R2, logistic_grad
from NNDebugger import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# remove 

# print(np.random.randn(2,3))	

class NeuralNetwork:
	"""NeuralNetwork
	X_data_full, y_data, n_layers, n_nodes_in_layer, n_epochs, batch_size, 
	n_hidden_layers: int, number of HIDDEN layers
	n_nodes_in_layer: array, node count for each layer, length must equal to n_hidden_layers
	n_catagotires: int, number of possible output, 0 for regression problems, 
	learning_rate, lmbd: regularisation params
	"""
	def __init__(
			self, 
			X_data_full, 
			y_data_full,  
			n_hidden_layers, 
			n_nodes_in_layer, 
			n_catagories=1,
			n_epochs=10, 
			batch_size=100, 
			learning_rate=0.001, 
			lmbd=0.01,
			activation="leaky_relu",
			activation_out="linear",
			is_classifier=False,
			is_debug=False
			):

		self.X_data_full = X_data_full
		self.layer_as = np.zeros(n_hidden_layers+1,dtype=object)# +1 for input to the output layer
		self.layer_zs = np.zeros(n_hidden_layers+1,dtype=object)

		self.y_data_full = y_data_full
		self.n_hidden_layers = n_hidden_layers
		self.n_nodes_in_layer = n_nodes_in_layer
		self.n_catagories = n_catagories

		self.n_inputs = X_data_full.shape[0]
		self.n_features = X_data_full.shape[1]
		

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.n_iter = self.n_inputs // self.batch_size


		self.learning_rate = learning_rate
		self.lmbd = lmbd

		self.activation = ActivationFunction(activation)
		self.activation_out = ActivationFunction(activation_out)

		self.is_classifier = is_classifier

		self.create_biases_and_weights()


		# debug
		self.debugger = NNDebugger(self, is_debug, activation, activation_out)

		self.debugger.print_static()

	# structural methods below

	def create_biases_and_weights(self):
		'''Initialise weights and biases vectors/matrices/tensors'''

		# weights and biases are both arrays of all other vectors/matrices with each entry coorresponds
		# to each layer
		self.weights = np.zeros(self.n_hidden_layers+1, dtype=object)
		self.biases = np.zeros(self.n_hidden_layers+1, dtype=object)

		self.weights[0] = np.random.randn(self.n_features, self.n_nodes_in_layer[0])
		self.biases[0] = np.zeros(self.n_nodes_in_layer[0]) + 0.01

		for i in range(1,self.n_hidden_layers):
			self.weights[i] = np.random.randn(self.n_nodes_in_layer[i-1], self.n_nodes_in_layer[i])
			self.biases[i] = np.zeros(self.n_nodes_in_layer[i]) + 0.01

		# output weights and bias
		self.weights[-1] = np.random.randn(self.n_nodes_in_layer[-1], self.n_catagories)
		self.biases[-1] = np.zeros(self.n_catagories) + 0.01

	# algorithmic methods below

	def feed_forward(self):

		self.layer_zs[0] = np.matmul(self.input, self.weights[0]) + self.biases[0]
		self.layer_as[0] = self.activation.func(self.layer_zs[0])

		for i in range(1,self.n_hidden_layers):
			# looping thru each layer updating all nodes

			self.layer_zs[i] = np.matmul(self.layer_as[i-1], self.weights[i]) + self.biases[i]
			self.layer_as[i] = self.activation.func(self.layer_zs[i])

		self.layer_zs[-1] = np.matmul(self.layer_as[-2], self.weights[-1]) + self.biases[-1]
		self.layer_as[-1] = self.activation_out.func(self.layer_zs[-1])
		
		# self.debugger.print_ff()


	def feed_forward_out(self, X):

		layer_zs = np.zeros(self.n_hidden_layers+1, dtype=object)
		layer_as = np.zeros(self.n_hidden_layers+1, dtype=object)

		layer_zs[0] = np.matmul(X, self.weights[0]) + self.biases[0]
		layer_as[0] = self.activation.func(layer_zs[0])

		for i in range(1,self.n_hidden_layers):
			# looping thru each layer updating all nodes
			# print(i)
			# print(self.layer_as[i].shape)
			# print(self.weights[i].shape)
			layer_zs[i] = np.matmul(layer_as[i-1], self.weights[i]) + self.biases[i]
			layer_as[i] = self.activation.func(layer_zs[i])


		layer_zs[-1] = np.matmul(layer_as[-2], self.weights[-1]) + self.biases[-1]
		layer_as[-1] = self.activation_out.func(layer_zs[-1])

		# print("out",output)

		return layer_as[-1] 

	def back_propagate(self):
		# initialise
		self.errors = np.zeros(self.n_hidden_layers+1, dtype=object)
		self.dws = np.zeros(self.n_hidden_layers+1, dtype=object)
		self.dbs = np.zeros(self.n_hidden_layers+1, dtype=object)

		# output layer
		self.errors[-1] = self.cost_grad(self.y_data, self.layer_as[-1])
		# print(self.errors[-1].shape)
		# print(self.layer_as[-1].shape)
		self.dws[-1] = np.matmul(self.layer_as[-2].T, self.errors[-1])
		self.dbs[-1] = np.sum(self.errors[-1], axis=0)
		

		for i in range(self.n_hidden_layers-1, -1, -1):

			# print("here", self.errors[i+1].shape, self.weights[i+1].T.shape)
			self.errors[i] = np.matmul(self.errors[i+1], self.weights[i+1].T) * self.activation.gradient(self.layer_as[i])


		self.dws[0] = np.matmul(self.input.T, self.errors[0])
		self.dbs[0] = np.sum(self.errors[0], axis=0)

		for i in range(1, self.n_hidden_layers):
			self.dws[i] = np.matmul(self.layer_as[i-1].T, self.errors[i])
			self.dbs[i] = np.sum(self.errors[i], axis=0)


		if self.lmbd > 0:
			for i in range(self.n_hidden_layers+1):
				self.dws[i] += self.lmbd * self.weights[i]

		self.weights -= self.learning_rate * self.dws
		self.biases -= self.learning_rate * self.dbs

	def train(self):
		data_indices = np.arange(self.n_inputs)

		for i in range(self.n_epochs):
			for j in range(self.n_iter):

				# for debugging
				curr_step = i*self.n_iter+j
				steps = self.n_epochs*self.n_iter

				# pick datapoints with replacement
				chosen_datapoints = np.random.choice(
					data_indices, size=self.batch_size, replace=False
				)

				# minibatch training data
				self.input = self.X_data_full[chosen_datapoints]
				self.y_data = self.y_data_full[chosen_datapoints]


				self.feed_forward()
				# self.debugger.print_ff(curr_step, steps)
				self.back_propagate() 
				# self.debugger.print_bp(curr_step, steps)

				# protection against overflow
				if self.score(self.X_data_full, self.y_data_full) > 100:
					self.create_biases_and_weights()

				self.debugger.print_score(i*self.n_iter+j,self.n_epochs*self.n_iter)

		training_score = self.score(self.X_data_full, self.y_data_full)
		if not self.is_classifier:
			if training_score > 0.5:
				print("CONVERGENCE ERROR: Maximum iteration (" + str(self.n_epochs*self.n_iter) 
					+ ") reached. Model has not converged, try increasing the number of iterations.")

	def prep(self):
		# prepare data to manual training

		data_indices = np.arange(self.n_inputs)
		chosen_datapoints = np.random.choice(
					data_indices, size=self.batch_size, replace=False)

		self.input = self.X_data_full[chosen_datapoints]
		self.y_data = self.y_data_full[chosen_datapoints]


	# Evaluation


class NNRegressor(NeuralNetwork):
	"""docstring for RegressionNeuralNetwork"""
	def __init__(self, 
			X_data_full, 
			y_data,  
			n_hidden_layers, 
			n_nodes, 
			n_epochs=100, 
			batch_size=1000, 
			learning_rate=0.01, 
			lmbd=0.0,
			activation="leaky_relu",
			activation_out="linear",
			is_debug=False):
		super(NNRegressor, self).__init__(X_data_full, y_data, n_hidden_layers, n_nodes,
			n_epochs=n_epochs, batch_size=batch_size, 
			learning_rate=learning_rate, 
			lmbd=lmbd,
			activation=activation, 
			activation_out=activation_out,is_debug=is_debug)
		self.cost_grad = MSE_prime


	def __repr__(self):
		# construct and return a string that represents the network
		# architecture
		return "NNRegressor: {}".format(str(self.n_nodes_in_layer))


	def predict(self,X):

		# check convergences

		return self.feed_forward_out(X)

	def R2(self,X,y):
		y_pred = self.predict(X)
		return R2(y,y_pred)


	def score(self, X, y):
		'''Evaluation of model'''
		y_pred = self.predict(X)
		return MSE(y, y_pred)


class NNClassifier(NeuralNetwork):
	"""docstring for RegressionNeuralNetwork"""
	def __init__(self, 
			X_data_full, 
			y_data,  
			n_hidden_layers, 
			n_nodes, 
			n_catagories,
			n_epochs=100, 
			batch_size=1000, 
			learning_rate=0.01, 
			lmbd=0.0,
			activation="leaky_relu",
			activation_out="linear",
			is_debug=False):
		super(NNClassifier, self).__init__(X_data_full, y_data, n_hidden_layers, n_nodes,
			n_catagories=n_catagories,n_epochs=n_epochs, batch_size=batch_size, 
			learning_rate=learning_rate, 
			lmbd=lmbd,
			activation=activation, 
			activation_out=activation_out,is_debug=is_debug, is_classifier=True)
		self.cost_grad = logistic_grad


	def __repr__(self):
		# construct and return a string that represents the network
		# architecture
		return "NNRegressor: {}".format(str(self.n_nodes_in_layer))


	def predict_prob(self,X):
		return self.feed_forward_out(X)

	def predict(self,X):
		probabilities = self.predict_prob(X)
		return np.where(probabilities>0.5, 1, 0)
		
	def score(self, X_test, y_test):

		y_pred = np.zeros(y_test.shape)
		y_pred[self.predict_prob(X_test) > 0.5] = 1

		return np.sum(y_pred == y_test) / len(y_test)





















