import numpy as np

class Sigmoid:
	def forward(self, x):
		return 1.0/(1+np.exp(x))
		
	def derivative_softmax(self, x, top_diff):
		output = self.forward(x)
		return (1.-output)*output*top_diff
	
class Tanh:
	def forward(self, x):
		return np.tanh(x)
		
	def derivative_tanh(self,x,top_diff):
		output = self.forward(x)
		return (1.-np.square(output))*top_diff
