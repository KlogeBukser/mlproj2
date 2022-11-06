# activation_funcs.py

# Sigmoid, the RELU and the Leaky RELU

def sigmoid(x):
	return 1/(1+np.exp(-x))


def relu():
	pass

def leaky_relu():
	pass

def heaviside(x):
	if x > 0:
		return 1
	return 0
