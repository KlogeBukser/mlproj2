from jax import numpy as jnp
import numpy as np
from activation_funcs import *
from gradient_descent import *

# print(np.random.randn(2,3))	

class NeuralNetwork:
	"""NeuralNetwork
	X_inputs, Y_inputs, n_layers, n_nodes, n_epochs, batch_size, 
	n_hidden_layers: int, number of HIDDEN layers
	n_nodes: array, node count for each layer, length must equal to n_hidden_layers
	n_catagotires: int, number of possible output, 0 for regression problems, 
	eta, lmbd: regularisation params
	"""
	def __init__(
			self, 
			X_inputs, 
			Y_inputs,  
			n_hidden_layers, 
			n_nodes, 
			gd_func,
			n_catagories=0, # only for classification
			n_epochs=10, 
			batch_size=100, 
			eta=0.01, 
			lmbd=0.0):
		super(NeuralNetwork, self).__init__()

		self.X_inputs = X_inputs
		self.Y_inputs = Y_inputs
		self.n_hidden_layers = n_hidden_layers
		self.n_nodes = n_nodes

		self.n_inputs = X_inputs.shape[0]
		self.n_features = X_inputs.shape[1]
		self.n_catagories = n_catagories

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.n_iter = self.n_inputs // self.batch_size


		self.eta = eta
		self.lmbd = lmbd

		self.gradient_descent = gd_func
		self.acti_func = acti_func
		self.acti_func_out = sigmoid

		self.create_biases_and_weights()


	# structural methods below

	def create_biases_and_weights(self):
		'''Initialise weights and biases vectors/matrices/tensors'''
		self.weightss = np.zeros(n_hidden_layers)
		for i in range(self.n_hidden_layers+1):
			continue



	# def regularisation(self):
	# 	pass


	# algorithmic methods below

	def feed_forward(self):
		pass

	def feed_forward_out(self):
		'''outputs the probabilities for each catagories'''
		pass


	def back_propagate(self):
		pass


	def predict(self):
		return 0

	def train(self):
		pass


	def acti_func(self):
		acti_func = sigmoid


	# Evaluation

	def score(self, X_test, Y_test):
		'''Evaluation of model'''
		Y_pred = self.predict(X_test)
		return np.sum(Y_pred == Y_test) / len(Y_test)
		

class NNRegressor(NeuralNetwork):
	"""docstring for RegressionNeuralNetwork"""
	def __init__(self):
		super(NNRegressor, self).__init__()
		


class NNClassifier(NeuralNetwork):
	"""docstring for Classifier"""
	def __init__(self):
		super(NNClassifier, self).__init__()
		
	


