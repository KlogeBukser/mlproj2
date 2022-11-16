# NNDebugger.py
import numpy as np

class NNDebugger:

	def __init__(self, mynn, is_debug, activation, activation_out):
		self.mynn = mynn
		self.is_debug = is_debug
		self.activation = activation
		self.activation_out = activation_out

	def print_static(self):
		if self.is_debug:
			print(" \n ======================Static Params=========================\n ")
			print("learning_rate", self.mynn.learning_rate)
			print("lmbd", self.mynn.lmbd)
			print("activation function:", self.activation)
			print("output activation function:", self.activation_out)
			print("n_epochs", self.mynn.n_epochs)
			print("Batch_size", self.mynn.batch_size)
			print(" \n ============================================================\n ")

	def print_wb(self, is_shape=True):
		if self.is_debug:
			print(" \n ======================Weights and Biases ========================\n ")
			if is_shape:
				for i in range(len(self.mynn.weights)-1):
					print("weights " + str(i), self.mynn.weights[i].T.shape)
					print("bias " + str(i), self.mynn.biases[i].shape)
				print("weights out", self.mynn.weights[-1].T.shape)
				print("bias out", self.mynn.biases[-1].T.shape)
			else:
				for i in range(len(self.mynn.weights)-1):
					print("weights " + str(i), self.mynn.weights[i].T)
					print("bias " + str(i), self.mynn.biases[i])
				print("weights out", self.mynn.weights[-1].T)
				print("bias out", self.mynn.biases[-1].T)
			print(" \n ============================================================\n ")


	def print_xy(self):
		if self.is_debug:
			print("x", self.mynn.input.T)
			print("y", self.mynn.y_data.T)


	def print_ff(self, curr_step=10, steps=10):
		if self.is_debug:
			if not curr_step % (steps/10):
				print(" \n ======================Feed Forward=========================\n\n ")
				print("activations",self.mynn.layer_as)
				print(" \n ============================================================\n ")


	def print_bp(self,curr_step=10,steps=10):

		if self.is_debug:
			if not curr_step % (steps/10):
				print(" \n\n ======================Back Propagate=========================\n ")
				count = 0
				for errors in self.mynn.errors:
					# print("errors " + str(count), self.mynn.errors[count].T)
					print("errors " + str(count), self.mynn.errors[count].T)
					count += 1
				# print("out_err", self.mynn.errors[-1].T)
				for i in range(len(self.mynn.dws)):

					# print("weights gradient" + str(i), self.mynn.dws[i])
					# print("biases gradient" + str(i), self.mynn.dbs[i])

					print("mean weights gradient" + str(i), np.mean(self.mynn.dws[i]))
					print("mean biases gradient" + str(i), np.mean(self.mynn.dbs[i]))
			print(" \n ============================================================\n ")

	def print_score(self, curr_step=10, steps=10):

		if self.is_debug:
			if not curr_step % (steps/10):
				print("iter", curr_step, "score", 
					self.mynn.score(self.mynn.X_data_full, self.mynn.y_data_full))

