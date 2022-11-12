from jax import numpy as jnp
import numpy as np
from activation_funcs import ActivationFunction
from gradient_descent import *
from misc import MSE, MSE_prime
from jax import grad
from NNDebugger import *

# test use
from sklearn.model_selection import train_test_split
from generate import gen_simple
import matplotlib.pyplot as plt

# print(np.random.randn(2,3))	

class NeuralNetwork:
	"""NeuralNetwork
	X_data_full, Y_data, n_layers, n_nodes_in_layer, n_epochs, batch_size, 
	n_hidden_layers: int, number of HIDDEN layers
	n_nodes_in_layer: array, node count for each layer, length must equal to n_hidden_layers
	n_catagotires: int, number of possible output, 0 for regression problems, 
	learning_rate, lmbd: regularisation params
	"""
	def __init__(
			self, 
			X_data_full, 
			Y_data_full,  
			n_hidden_layers, 
			n_nodes_in_layer, 
			n_catagories=1,
			n_epochs=10, 
			batch_size=100, 
			learning_rate=0.001, 
			lmbd=0.01,
			activation="leaky_relu",
			activation_out="linear",
			is_debug=False
			):

		self.X_data_full = X_data_full
		self.layer_as = np.zeros(n_hidden_layers+1,dtype=object)# +1 for input to the output layer
		self.layer_zs = np.zeros(n_hidden_layers+1,dtype=object)

		self.Y_data_full = Y_data_full
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

		self.create_biases_and_weights()


		# debug
		self.debugger = NNDebugger(self, is_debug)

		


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
			# print(i)
			# print(self.layer_as[i].shape)
			# print(self.weights[i].shape)
			self.layer_zs[i] = np.matmul(self.layer_as[i-1], self.weights[i]) + self.biases[i]
			self.layer_as[i] = self.activation.func(self.layer_zs[i])
			# print(self.layer_as[i+1].shape)

		self.layer_zs[-1] = np.matmul(self.layer_as[-2], self.weights[-1]) + self.biases[-1]
		self.layer_as[-1] = self.activation_out.func(self.layer_zs[-1])
		# print("output shape is", self.output.shape )
		# print("data shape is", self.Y_data.shape)
		# print("out",np.mean(self.output.T))


	def feed_forward_out(self, X):

		layer_zs = np.zeros(self.n_hidden_layers+1, dtype=object)
		layer_as = np.zeros(self.n_hidden_layers+1, dtype=object)

		layer_zs[0] = np.matmul(X, self.weights[0]) + self.biases[0]
		layer_as[0] = self.activation.func(layer_zs[0])

		for i in range(1,self.n_hidden_layers+1):
			# looping thru each layer updating all nodes
			# print(i)
			# print(self.layer_as[i].shape)
			# print(self.weights[i].shape)
			layer_zs[i] = np.matmul(layer_as[i-1], self.weights[i]) + self.biases[i]
			layer_as[i] = self.activation.func(layer_zs[i])

		# layer_inputs = np.zeros(self.n_hidden_layers+1, dtype=object)
		# layer_inputs[0] = X

		# for i in range(self.n_hidden_layers):
		# 	z = np.matmul(layer_inputs[i], self.weights[i]) + self.biases[i]
		# 	layer_inputs[i+1] = self.activation.func(z)


		# output = np.matmul(layer_inputs[-1], self.weights[-1]) + self.biases[-1]
		# print("out",output)

		return layer_zs[-1] 


	def cost(self): # square cost function
		# never actually used, just to show
		return 0.5*(self.output-self.Y_data)**2

	def back_propagate(self):
		# initialise
		self.errors = np.zeros(self.n_hidden_layers+1, dtype=object)
		self.dws = np.zeros(self.n_hidden_layers+1, dtype=object)
		self.dbs = np.zeros(self.n_hidden_layers+1, dtype=object)

		# output layer
		self.errors[-1] = MSE_prime(self.Y_data, self.layer_as[-1])
		# self.errors[-1] = MSE_prime(self.layer_as[-1],self.Y_data) * self.activation.gradient(self.layer_as[-1])
		# print(self.errors[-1].shape)
		# print(self.layer_as[-2].shape)
		self.dws[-1] = np.matmul(self.layer_as[-2].T, self.errors[-1])
		self.dbs[-1] = np.sum(self.errors[-1], axis=0)
		if self.lmbd > 0:
			self.dws[-1] += self.lmbd * self.weights[-1]

		for i in range(self.n_hidden_layers-1, -1, -1):
			# delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			# nabla_b[-l] = delta
			# nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

			# print(self.errors[i+1].shape)
			# print(self.weights[i+1].T.shape)
			self.errors[i] = np.matmul(self.errors[i+1], self.weights[i+1].T) * self.activation.gradient(self.layer_zs[i])
			self.dws[i] = np.matmul(self.layer_as[i-1].T, self.errors[i])

			self.dbs[i] = np.sum(self.errors[i], axis=0)

			if self.lmbd > 0:
				self.dws[i] += self.lmbd * self.weights[i]

		# print("dws",self.dws[-1].shape)

		# print(self.weights.shape == self.dws.shape)
		self.weights -= self.learning_rate * self.dws
		self.biases -= self.learning_rate * self.dbs



	def train(self):
		data_indices = np.arange(self.n_inputs)

		for i in range(self.n_epochs):
			for j in range(self.n_iter):
				# pick datapoints with replacement
				chosen_datapoints = np.random.choice(
					data_indices, size=self.batch_size, replace=False
				)

				# minibatch training data
				self.input = self.X_data_full[chosen_datapoints]
				self.Y_data = self.Y_data_full[chosen_datapoints]

				self.feed_forward()
				self.back_propagate() 

				self.debugger.print_score(i*self.n_iter+j,self.n_epochs*self.n_iter)

	def prep(self):
		# no random shit

		data_indices = np.arange(self.n_inputs)
		chosen_datapoints = np.random.choice(
					data_indices, size=self.batch_size, replace=False)

		self.input = self.X_data_full[chosen_datapoints]
		self.Y_data = self.Y_data_full[chosen_datapoints]


	# Evaluation


class NNRegressor(NeuralNetwork):
	"""docstring for RegressionNeuralNetwork"""
	def __init__(self, 
			X_data_full, 
			Y_data,  
			n_hidden_layers, 
			n_nodes, 
			n_epochs=100, 
			batch_size=1000, 
			learning_rate=0.01, 
			lmbd=0.0,
			activation="leaky_relu",
			activation_out="linear",
			is_debug=False):
		super(NNRegressor, self).__init__(X_data_full, Y_data, n_hidden_layers, n_nodes,
			n_epochs=n_epochs, batch_size=batch_size, 
			learning_rate=learning_rate, 
			lmbd=lmbd,
			activation=activation, 
			activation_out=activation_out,is_debug=is_debug)


	def __repr__(self):
		# construct and return a string that represents the network
		# architecture
		return "NNRegressor: {}".format(str(self.n_nodes_in_layer))

	def predict(self,X):

		return self.feed_forward_out(X)

	def score(self, X_test, Y_test):
		'''Evaluation of model'''
		Y_pred = self.predict(X_test)
		return MSE(Y_test, Y_pred)

	

class NNClassifier(NeuralNetwork):
	"""docstring for Classifier"""
	def __init__(self, 
			X_data_full, 
			Y_data,  
			n_hidden_layers, 
			n_nodes, 
			n_catagories,
			acti_func_out="sigmoid",
			n_epochs=10, 
			batch_size=100, 
			learning_rate=0.01, 
			lmbd=0.0,
			):
		super(NNClassifier, self).__init__(X_data_full, Y_data, n_hidden_layers, n_nodes,
			n_epochs, batch_size, learning_rate, lmbd)
		self.n_catagories = n_catagories
		self.out_biases = np.zeros(self.n_catagories) + 0.01
		self.activation_out = ActivationFunction(acti_func_out)

	def __repr__(self):
		# construct and return a string that represents the network
		# architecture
		return "NNClassifier: {}".format("-".join(str(self.n_nodes_in_layer)))

	def predict(self,X):

		probabilities = self.feed_forward_out(X)
		return np.argmax(probabilities, axis=1)

	def score(self, X_test, Y_test):
		'''Evaluation of model'''
		Y_pred = self.predict(X_test)
		return np.sum(Y_pred == Y_test) / len(Y_test)

















