from jax import numpy as jnp
import numpy as np
from activation_funcs import ActivationFunction
from gradient_descent import *
from misc import MSE
from jax import grad

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
			gd_func,
			n_epochs=10, 
			batch_size=100, 
			learning_rate=0.01, 
			lmbd=0.0):

		self.X_data_full = X_data_full
		self.layer_inputs = np.zeros(n_hidden_layers+1,dtype=object)# +1 for input to the output layer

		self.Y_data_full = Y_data_full
		self.n_hidden_layers = n_hidden_layers
		self.n_nodes_in_layer = n_nodes_in_layer

		self.n_inputs = X_data_full.shape[0]
		self.n_features = X_data_full.shape[1]
		

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.n_iter = self.n_inputs // self.batch_size


		self.learning_rate = learning_rate
		self.lmbd = lmbd

		self.gradient_descent = gd_func
		self.activation = ActivationFunction("sigmoid")

		self.create_biases_and_weights()
		


	# structural methods below

	def create_biases_and_weights(self):
		'''Initialise weights and biases vectors/matrices/tensors'''

		# weights and biases are both arrays of all other vectors/matrices with each entry coorresponds
		# to each layer
		self.weights = np.zeros(self.n_hidden_layers, dtype=object)
		self.biases = np.zeros(self.n_hidden_layers, dtype=object)

		self.weights[0] = np.random.randn(self.n_features, self.n_nodes_in_layer[0])

		for i in range(1,self.n_hidden_layers):
			self.weights[i] = np.random.randn(self.n_nodes_in_layer[i-1], self.n_nodes_in_layer[i])
			self.biases[i] = np.zeros(self.n_nodes_in_layer[i]) + 0.01

		self.out_weights = np.random.randn(self.n_nodes_in_layer[-1],self.n_features)
		self.out_biases = 0.01

	# algorithmic methods below

	def feed_forward(self):

		for i in range(self.n_hidden_layers):
			# looping thru each layer updating all nodes
			# print(i)
			# print(self.layer_inputs[i].shape)
			# print(self.weights[i].shape)
			z = np.matmul(self.layer_inputs[i], self.weights[i]) + self.biases[i]
			self.layer_inputs[i+1] = self.activation.func(z)
			# print(self.layer_inputs[i+1].shape)

		self.output = np.matmul(self.layer_inputs[-1], self.out_weights) + self.out_biases

		# print("output shape is", self.output.shape )
		# print("data shape is", self.Y_data.shape)
		print("out",self.output)


	def feed_forward_out(self, X):

		layer_inputs = np.zeros(self.n_hidden_layers+1, dtype=object)
		layer_inputs[0] = X

		for i in range(self.n_hidden_layers):
			z = np.matmul(layer_inputs[i], self.weights[i]) + self.biases[i]
			layer_inputs[i+1] = self.activation.func(z)


		output = np.matmul(layer_inputs[-1], self.out_weights) + self.out_biases
		print("out",output)

		return output


	def cost(self): # square cost function
		# never actually used, just to show
		return 0.5*(self.output-self.Y_data)**2

	def back_propagate(self):

		# problematic

		# out_err = (self.output - self.Y_data) * self.activation_out.gradient(self.layer_inputs[-1])
		out_err = self.output - self.Y_data
		hidden_errors = np.zeros(self.n_hidden_layers, dtype=object)
		hidden_errors[-1] = np.matmul(out_err, self.out_weights.T) * self.activation.gradient(self.layer_inputs[-1]) # * self.layer_inputs[i] * (1 - self.layer_inputs[i])

		# debug
		# print("out err", out_err.T, "\n", out_err.shape)
		# print("hidden err", hidden_errors)#,"\n", hidden_errors[-1].shape)
		# print("weight", self.out_weights.T, "\n", self.out_weights.T.shape)

		self.hidden_weights_gradients = np.zeros(self.n_hidden_layers, dtype=object)
		self.hidden_bias_gradients = np.zeros(self.n_hidden_layers, dtype=object)

		for i in range(2, self.n_hidden_layers+1):

			ind_curr = -i
			hidden_errors[ind_curr] = np.matmul(hidden_errors[ind_curr+1], 
				self.weights[ind_curr+1].T) * self.activation.gradient(
				self.layer_inputs[ind_curr])


		# update output gradients
		self.out_weights_gradient = np.matmul(self.layer_inputs[-1].T, out_err)
		self.output_bias_gradient = np.sum(out_err, axis=0)

		# update hidden gradients
		for i in range(self.n_hidden_layers):
			self.hidden_weights_gradients[i] = np.matmul(self.layer_inputs[i].T, hidden_errors[i])
			self.hidden_bias_gradients[i] = np.sum(hidden_errors[i], axis=0)

			# update gradients with regularisation params
			if self.lmbd > 0.0:
				self.out_weights_gradients[i] += self.lmbd * self.out_weights
				self.hidden_weights_gradients[i] += self.lmbd * self.weights

		
		# update weights and bias
		self.out_weights -= self.learning_rate * self.out_weights_gradient
		self.out_biases -= self.learning_rate * self.output_bias_gradient
		for i in range(self.n_hidden_layers):
			self.weights[i] -= self.learning_rate * self.hidden_weights_gradients[i]
			self.biases[i] -= self.learning_rate * self.hidden_bias_gradients[i]


	def train(self):
		data_indices = np.arange(self.n_inputs)

		for i in range(self.n_epochs):
			for j in range(self.n_iter):
				# pick datapoints with replacement
				chosen_datapoints = np.random.choice(
					data_indices, size=self.batch_size, replace=False
				)

				# minibatch training data
				self.layer_inputs[0]= self.X_data_full[chosen_datapoints]
				self.Y_data = self.Y_data_full[chosen_datapoints]

				self.feed_forward()
				self.back_propagate() 

	def train2(self):
		# no random shit

		for i in range(1000):
			# just 1000 iteration
			self.layer_inputs[0] = self.X_data_full
			self.Y_data = self.Y_data_full

			self.feed_forward()
			self.back_propagate() 


	# Evaluation


class NNRegressor(NeuralNetwork):
	"""docstring for RegressionNeuralNetwork"""
	def __init__(self, 
			X_data_full, 
			Y_data,  
			n_hidden_layers, 
			n_nodes, 
			gd_func,
			n_catagories,
			n_epochs=10, 
			batch_size=100, 
			learning_rate=0.01, 
			lmbd=0.0,):
		super(NNRegressor, self).__init__(X_data_full, Y_data, n_hidden_layers, n_nodes, gd_func,
			n_epochs=10, batch_size=100, learning_rate=0.01, lmbd=0.0)

	def __repr__(self):
		# construct and return a string that represents the network
		# architecture
		return "NNRegressor: {}".format("-".join(str(self.n_nodes_in_layer)))

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
			gd_func,
			n_catagories,
			acti_func_out="sigmoid",
			n_epochs=10, 
			batch_size=100, 
			learning_rate=0.01, 
			lmbd=0.0,
			):
		super(NNClassifier, self).__init__(X_data_full, Y_data, n_hidden_layers, n_nodes, gd_func, 
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