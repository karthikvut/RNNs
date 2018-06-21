from datetime import datetime
import numpy as np
import sys
from layers import RNNLayer
from output import Softmax

class Model:
	
	def __init__(self, word_dim, hidden_dim=10, bptt_truncate=4):
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		self.U = np.random.uniform(-np.sqrt(1./ word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
		
	def forward_propagation(self, x):
		
		T = len(x)
		layers_ = []
		prev_state = np.zeros(self.hidden_dim)
		for t in range(T):
			layer = RNNLayer()
			input = np.zeros(self.word_dim)
			input[x[t]] = 1
			layer.forward(input, prev_state, self.U, self.W, self.V)
			prev_state = layer.state
			layers.append(layer)
		return layers
		
	def predict(self, x):
		output = Softmax()
		layers = self.forward_propagation(x)
		return [np.argmax(output.predict(layer.mulV)) for layer in layers]
		
	
	def calculate_loss(self, X, y):
		assert len(X) == len(y)
		output = Softmax()
		layers = self.forward_propagation(X)
		loss = 0.0
		for i, layer in enumerate(layers):
			loss += output.loss(layer.mulV, y[i])
		return loss/ float(len(y))
	
	def calculate_total_loss(self, X, y):
		loss = 0.
		for i in range(len(y)):
			loss += self.calculate_loss(X[i], y[i])
		return loss/float(len(y))
		
	def bptt(self, X, y):
		assert len(X) == len(y)
		output = Softmax()
		layers = self.forward_propagation(X)
		dU = np.zeros(self.U.shape)
		dV = np.zeros(self.V.shape)
		dW = np.zeros(self.W.shape)
		
		T = len(layers)
		prev_state_t = np.zeros(self.hidden_dim)
		diff_state = np.zeros(self.hidden_dim)
		for t in range(0, T):
			dmulV = output.diff(layers[t].mulV, y[t])
			input = np.zeros(self.word_dim)
			input[x[t]] = 1
			dprev_state, dU_t, dV_t = layers[t].backward(input, prev_state, self.U, self.W, self.V, diff_state, dmulV)
			prev_state_t = layers[t].state
			dmulV = np.zeros(self.word_dim)
			for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
				input = np.zeros(self.word_dim)
				input[x[t]] = 1
				prev_state_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].state
				dprev_state, dU_i, dW_i, dV_i = layers[i].backward(input, prev_state_i, self.U, self.W, self.V, dprev_state, dmulV)
				dU_t += dU_i
				dW_t += dW_i
			dV += dV_t
			dW += dW_t
			dU += dU_t
			
		return (dU, dW, dV)
		
	def sgd_step(self, x, y, learning_rate):
		dU, dW, dV = self.bptt(x, y)
		self.U -= learning_rate * dU
		self.W -= learning_rate * dW
		self.V -= learning_rate * dV		
	
	def train(self,X, y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
		num_examples_seen = 0
		losses = []
		for epoch in range(nepoch):
			if (epoch % evaluate_loss_after == 0):
				loss = self.calculate_loss(X, y)
				losses.append((num_examples_seen, loss))
				time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				print('%s: Loss after num_examples_seen=%d epoch=%d: %f" (time, num_examples_seen, epoch, loss))
				#Adjust learning rate if loss increases
				if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
					learning_rate = learning_rate * 0.5
					print("Setting learning rate to %f" % learning_rate)
				sys.stdout.flush()
			#For each training example
			for i in range(len(y)):
				self.sgd_step(X[i], y[i], learning_rate)
				num_examples_seen += 1
		return losses
