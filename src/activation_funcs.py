# activation_funcs.py
from jax import grad,vjp
import jax.numpy as np

# Sigmoid, the RELU and the Leaky RELU

# jax.grad requires function to have float output, hence the 0.0's, allow_int=True doesn't work

class ActivationFunction:
	"""Activation functions and their gradients"""
	def __init__(self, func, gradient=None):
		self.func = func

		if gradient:
			self.gradient = gradient
		else:
			self.auto_grad_it()

	def auto_grad():
		self.gradient = grad(self.func)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	if x > 0:
		return x
	return 0.0

def leaky_relu(x, alpha):
	if x >= 0:
		return x 
	return alpha*x

def heaviside(x):
	if x > 0:
		return 1.0
	return 0.0



