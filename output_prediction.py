import numpy as np

class Softmax:
	
	def predict(self, x):
		exp_x = np.exp(x)
		return exp_x/np.sum(exp_x)
		
	def loss(self, x, y):
		probs = self.predict(x)
		return -np.log(probs[y])
		
	def diff(self, x, y):
		probs = self.predict(x)
		probs[y] -= 1.0
		return probs
