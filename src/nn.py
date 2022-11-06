from jax import numpy as jnp
import numpy as np

# print(np.random.randn(2,3))
class Node:
	"""docstring for Node
	weights: array-like, length equal to the number of nodes in the previous layer
	biases: float, """
	def __init__(self, n_node_in_next_layer):
		super(Node, self).__init__()

		# Do NOT initialise to 0
		self.weights = np.ones(n) # small, (0,1), normally distribution 
		self.bias = np.random.randn(1) # small, (0,1), normally distribution


		

class NeuralNetwork:
	"""docstring for NeuralNetwork
	X_inputs, Y_inputs, n_layers, n_nodes, n_epochs, batch_size, 
	eta, lmbd: regularisation params
	"""
	def __init__(self, X_inputs, Y_inputs, n_catagories, n_layers, n_nodes, n_epochs=10, batch_size=100, eta=0.01, lmbd=0.0):
		super(NeuralNetwork, self).__init__()

		self.X_inputs = X_inputs
		self.Y_inputs = Y_inputs
		self.n_layers = n_layers
		self.n_nodes = n_nodes

		self.n_inputs = X_inputs.shape[0]
		self.n_features = X_inputs.shape[1]
		self.n_catagories = n_catagories

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.n_iter = self.n_inputs // self.batch_size


	# structural methods below

	def create_biases_and_weights(self):
		pass

	def add_node(self, node, ind_layer, ind_node):
		pass

	def cost_func(self):
		pass


	# def gradient_desent(self):
	# 	pass


	# def regularisation(self):
	# 	pass


	# algorithmic methods below

	def feed_forward(self):
		pass

	def feed_forward_out(self):
		pass


	def back_propagate(self):
		pass


	def predict(self):
		return 0

	def train(self):
		pass


	# Evaluation

	def score(self, X_test, Y_test):
	'''Evaluation of model'''
		Y_pred = self.predict(X_test)
		return np.sum(Y_pred == Y_test) / len(Y_test)




	


