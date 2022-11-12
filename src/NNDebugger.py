# NNDebugger.py
import numpy as np

class NNDebugger:

	def __init__(self, mynn, is_debug):
		self.mynn = mynn
		self.is_debug = is_debug

	def print_static(self):
		if self.is_debug:
			print(" \n ======================Static Params=========================\n\n ")
			print("learning_rate", self.mynn.learning_rate)
			print("lmbd", self.mynn.lmbd)

	def print_wb(self):
		if self.is_debug:
			print("weights out", self.mynn.weights[-1].T)
			print("weights 0", self.mynn.weights[0].T)
			print("weights 1", self.mynn.weights[1].T)
			print("bia out", self.mynn.biases[-1].T)
			print("bia 0", self.mynn.biases[0])
			print("bia 1", self.mynn.biases[1])


	def print_xy(self):
		if self.is_debug:
			print("x", self.mynn.input.T)
			print("y", self.mynn.Y_data.T)


	def print_ff(self):
		if self.is_debug:
			print(" \n ======================Feed Forward=========================\n\n ")
			print("activations",self.mynn.layer_as)


	def print_bp(self):

		if self.is_debug:
			print(" \n\n ======================Back Propagate=========================\n ")
			count = 0
			for errors in self.mynn.errors:
				# print("errors " + str(count), self.mynn.errors[count].T)
				print("errors " + str(count), np.mean(self.mynn.errors[count].T))
				count += 1
			# print("out_err", self.mynn.errors[-1].T)
			for i in range(len(self.mynn.dws)):

				# print("weights gradient" + str(i), self.mynn.dws[i])
				# print("biases gradient" + str(i), self.mynn.dbs[i])

				print("weights gradient" + str(i), np.mean(self.mynn.dws[i]))
				print("biases gradient" + str(i), np.mean(self.mynn.dbs[i]))

	def print_score(self, i, steps):

		if self.is_debug:
			if not i % (steps/10):
				print("iter", i, "score", 
					self.mynn.score(self.mynn.X_data_full, self.mynn.Y_data_full))

