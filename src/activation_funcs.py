# activation_funcs.py
from jax import grad,vjp
import numpy as np

# Sigmoid, the RELU and the Leaky RELU

# jax.grad requires function to have float output, hence the 0.0's, allow_int=True doesn't work

class ActivationFunction:
	"""Activation functions and their gradients"""
	def __init__(self, func_name):

		if func_name == "sigmoid":
			self.func = sigmoid
			self.gradient = sigmoid_grad

		elif func_name == "relu":
			self.func = relu
			self.gradient = relu_grad

		elif func_name == "leaky_relu":
			self.func = leaky_relu
			self.gradient = leaky_relu_func

		else:
			raise Exception("No matching function for name" + func_name)
	

def sigmoid(x):
	# print(x)
	return 1/(1+np.exp(-x))

def sigmoid_grad(x):
	return sigmoid(x)*(1-sigmoid(x))


def relu(x):
	
	if type(x) is int or type(x) is float:
		if x > 0:
			return x
		return 0.0

	result = np.zeros(len(x))
	for i in range(len(x)):
		if x[i] > 0:
			result[i] = x[i]

	return result

def relu_grad(x):

	if type(x) is int or type(x) is float:
		if x > 0:
			return 1
		return 0.0

	result = np.zeros(len(x))
	for i in range(len(x)):
		if x[i] > 0:
			result[i] = 1

	return result

def leaky_relu(x, alpha):

	if type(x) is int or type(x) is float:
		if x > 0:
			return x
		return alpha*x

	result = np.zeros(len(x))
	for i in range(len(x)):
		if x[i] > 0:
			result[i] = x[i]
		else:
			result[i] = alpha*x[i]

	return result

def leaky_relu_grad(x, alpha):

	if type(x) is int or type(x) is float:
		if x > 0:
			return 1
		return alpha

	result = np.zeros(len(x))
	for i in range(len(x)):
		if x[i] > 0:
			result[i] = 1
		else:
			result[i] = alpha

	return result


def heaviside(x):

	if type(x) is int or type(x) is float:
		if x > 0:
			return 1.0
		return 0.0

	result = np.zeros(len(x))
	if x > 0:
		result[i] = 1.0
	
	return result

# sig = ActivationFunction("sigmoid")
# a = np.array([1.0,2.0,3.0,-1])
# print(sig.func(a))
# print(sig.gradient(a))

# print(leaky_relu([1.2,-2.1],0.1))


