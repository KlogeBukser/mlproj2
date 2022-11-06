from jax import numpy as jnp
import numpy as np
from activation_funcs import *
from gradient_descent import *

# print(np.random.randn(2,3))	

class NeuralNetwork:
	"""NeuralNetwork
	X_inputs, Y_inputs, n_layers, n_nodes_in_layer, n_epochs, batch_size, 
	n_hidden_layers: int, number of HIDDEN layers
	n_nodes_in_layer: array, node count for each layer, length must equal to n_hidden_layers
	n_catagotires: int, number of possible output, 0 for regression problems, 
	eta, lmbd: regularisation params
	"""
	def __init__(
			self, 
			X_inputs, 
			Y_inputs,  
			n_hidden_layers, 
			n_nodes_in_layer, 
			gd_func,
			n_epochs=10, 
			batch_size=100, 
			eta=0.01, 
			lmbd=0.0):

		self.X_inputs = X_inputs
		self.layer_inputs = X_inputs # updates as we loop thru layers
		self.Y_inputs = Y_inputs
		self.n_hidden_layers = n_hidden_layers
		self.n_nodes_in_layer = n_nodes_in_layer

		self.n_inputs = X_inputs.shape[0]
		self.n_features = X_inputs.shape[1]
		

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.n_iter = self.n_inputs // self.batch_size


		self.eta = eta
		self.lmbd = lmbd

		self.gradient_descent = gd_func
		self.acti_func = acti_func
		self.acti_func_out = sigmoid

		self.create_biases_and_weights()
		self.create_bias_and_weights_out()


	# structural methods below

	def create_biases_and_weights(self):
		'''Initialise weights and biases vectors/matrices/tensors'''

		# weights and biases are both arrays of all other vectors/matrices with each entry coorresponds
		# to each layer
		self.weights = np.zeros(n_hidden_layers)
		self.biases = np.zeros(n_hidden_layers)

		for i in range(self.n_hidden_layers):
			self.weights[i] = np.random.randn(self.n_features, self.n_nodes_in_layer[i])
			self.biases[i] = np.zeros(n_nodes_in_layer[i]) + 0.01

		self.out_weights = np.random.randn(self.n_features, self.n_nodes_in_layer[i])
		self.out_biases = 0.01


	# def regularisation(self):
	# 	pass


	# algorithmic methods below

	def feed_forward(self):

		for i in range(self.n_hidden_layers):
			# looping thru each layer updating all nodes
			z = np.matmul(self.layer_inputs, self.weights[i]) + self.biases[i]
			self.layer_inputs = self.activation.func(z)

		self.z_o = np.matmul(self.layer_inputs, self.out_weights) + self.out_biases


	def feed_forward_out(self):
		'''outputs the probabilities for each catagories, only called when the given criterion are satisfied'''

		# no activation function for regression problems, or a = z
		return self.z_o


	def back_propagate(self):
		error_output = self.output - self.Y_data


	def predict(self):

		return 0

	def train(self):
		pass


	def activation(self,z):
		activation = ActivationFunction(sigmoid)


	# Evaluation

	def score(self, X_test, Y_test):
		'''Evaluation of model'''
		Y_pred = self.predict(X_test)
		return np.sum(Y_pred == Y_test) / len(Y_test)


class NNRegressor(NeuralNetwork):
	"""docstring for RegressionNeuralNetwork"""
	def __init__(self):
		super(NNRegressor, self).__init__(
			X_inputs, 
			Y_inputs,  
			n_hidden_layers, 
			n_nodes, 
			gd_func,
			n_epochs=10, 
			batch_size=100, 
			eta=0.01, 
			lmbd=0.0)
		


class NNClassifier(NeuralNetwork):
	"""docstring for Classifier"""
	def __init__(self, X_inputs, 
			Y_inputs,  
			n_hidden_layers, 
			n_nodes, 
			gd_func,
			n_catagories
			n_epochs=10, 
			batch_size=100, 
			eta=0.01, 
			lmbd=0.0,
			):
		super(NNClassifier, self).__init__(X_inputs, Y_inputs, n_hidden_layers, n_nodes, gd_func, 
			n_epochs, batch_size, eta, lmbd)
		self.n_catagories = n_catagories
		self.out_biases = np.zeros(self.n_catagories) + 0.01
		

	def activation_out(self, func=sigmoid):
		self.activation_out = ActivationFunction(func)
	
	def feed_forward_out(self):
		'''outputs the probabilities for each catagories'''

		probs = self.activation_out.func(self.z_o)

		return probs