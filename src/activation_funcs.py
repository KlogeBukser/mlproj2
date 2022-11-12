# activation_funcs.py
import numpy as np
import matplotlib.pyplot as plt
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
			self.gradient = leaky_relu_grad

		elif func_name == "linear":
			self.func = linear
			self.gradient = linear_grad

		else:
			raise Exception("No matching function for name" + func_name)
	

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_grad(x):
	return sigmoid(x)*(1-sigmoid(x))

def relu(x):
	return np.maximum(X, 0, out=X)
    
def relu_grad(x): 
    return 1 * (x > 0)

def leaky_relu(x, alpha=0.01):

	output = np.where(x > 0, x, x * alpha)

	return output

def leaky_relu_grad(x, alpha=0.01):

	output = np.where(x > 0, 1, alpha)

	return output

def linear(x):
	return x


def linear_grad(x):
	return np.ones(x.shape)


# sig = ActivationFunction("sigmoid")
# a = np.array([1.0,2.0,3.0,-1])
# print(sig.func(a))
# print(sig.gradient(a))
# li = ActivationFunction("linear")
# a = np.zeros((2,3))
# print(li.gradient(a))

# print(leaky_relu([1.2,-2.1],0.1))

# a=np.arange(-10,10,0.01)
# plt.plot(a, sigmoid(a))
# plt.plot(a,sigmoid_grad(a))
# plt.show()