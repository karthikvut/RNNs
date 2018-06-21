class RnnLayer:
	
	def forward(self, x, prev_state, U, W, V):
		self.mulU = mulGate.forward(U, x)
		self.mulW = mulGate.forward(W, prev_state)
		self.add = addGate.forward(self.mulU, self.mulW)
		self.state = activation.forward(self.add)
		self.mulV = mulGate.forward(V, self.state)
	
  
	def backward(self, x, prev_state, U, W, V, diff_state, dmulV):
	
    self.forward(x, prev_state, U, W, V)
		dV, dSV = mulGate.backward(V, self.state, dmulV)
		dS = dSV+diff_state
		dadd = activation.backward(self.add, dS)
		dmulW, dmulU = addGate.backward(self.mulU, self.mulW, dadd)
		dW, dprev_state = mulGate.backward(W, prev_state, dmulW)
		dU, dx = mulGate.backward(U, x, dmulU)
		return (dprev_state, dU, dW, dV)
