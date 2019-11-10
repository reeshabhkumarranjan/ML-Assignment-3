import numpy as np


class NeuralNet:
	num_layers = None
	num_nodes = None
	activation_function = None
	learning_rate = None
	weights = None
	outputs = None
	outputs_derivative = None
	deltas = None

	def __init__(self, num_layers, num_nodes, activation_function, learning_rate):
		self.num_layers = num_layers  # number of layers including input and output
		self.num_nodes = num_nodes  # list of number of nodes in each layer
		self.activation_function = activation_function  # activation function to be used (string)
		self.learning_rate = learning_rate  # learning rate
		self.weights = [None] * (num_layers - 1)  # it is a list of list of numpy arrays
		# the [i][j] index of the data-structure corresponds to weight to the jth node in the
		# (i + 1)th layer from all the nodes in the ith layer.
		self.outputs = [None] * (num_layers)  # it is a list of numpy arrays
		# it store the output of all the layers, where output is the phi(v)
		self.outputs_derivative = [None] * (num_layers)  # it is a list of numpy arrays
		# it stores the derivative of output of all the layers, where the derivative is phi'(v)
		self.deltas = [None] * (num_layers)  # it is a list of numpy arrays.
		# it stores delta values corresponding to each node in a given later.

		for layer in range(num_layers - 1):
			self.weights[layer] = [None] * (num_nodes[layer + 1])
			for node in range(num_nodes[layer + 1]):
				self.weights[layer][node] = np.random.normal(loc=0, scale=1, size=(num_nodes[layer] + 1, 1))
				self.weights[layer][node][-1] = 0

		for output in range(len(self.outputs)):
			self.outputs[output] = np.empty((num_nodes[output], 1))
			self.deltas[output] = np.empty((num_nodes[output], 1))

	def forward_phase(self, input):
		input = input.reshape(-1, 1)
		self.outputs[0] = input
		self.outputs_derivative[0] = Relu().grad(self.outputs[0])
		for layer in range(1, self.num_layers):
			for node in range(self.num_nodes[layer]):
				input = np.concatenate((self.outputs[layer - 1], np.ones((1, 1))))
				output = np.dot(np.transpose(input), self.weights[layer - 1][node])
				output = Relu().value(output)
				self.outputs[layer][node] = output
			self.outputs_derivative[layer] = Relu().grad(self.outputs[layer])

	def backward_phase(self, d, layer=0):
		"""Call it with layer = 1"""

		if layer == self.num_layers - 2:
			for node in range(self.num_nodes[layer + 1]):
				error_signal = d[node] - self.outputs[layer + 1][node]
				phi_dash = self.outputs_derivative[layer + 1][node]  # TODO is this correct?
				delta = error_signal * phi_dash
				self.deltas[layer + 1][node] = delta

				# adjust weights connecting to this node

				for previous_node in range(self.num_nodes[layer]):
					w_delta = self.learning_rate * self.deltas[layer + 1][node] * self.outputs[layer][previous_node]
					self.weights[layer][node][previous_node] -= w_delta
				return

		# first make sure that the delta values for the next layer are availabke
		self.backward_phase(d=d, layer=layer + 1)

		# now start adjusting the weights emerging from every node in the current layer
		for node in range(self.num_nodes[layer]):

			# calculate the delta sum using the delta values of nodes in the next layer
			delta_sum = 0
			for next_node in range(self.num_nodes[layer + 1]):
				delta_sum += self.deltas[layer + 1][next_node] * self.weights[layer][next_node][node]
			phi_dash = self.outputs_derivative[layer][node]
			delta = delta_sum * phi_dash
			self.deltas[layer][node] = delta
			for next_node in range(self.num_nodes[layer + 1]):
				w_delta = self.learning_rate * self.deltas[layer + 1][next_node] * self.outputs[layer][node]
				self.weights[layer][next_node][node] -= w_delta
		return

	def fit(self, X, Y, batch_size, epochs):
		pass

	def predict(self, X):
		pass

	def score(self, X, Y):
		pass


class Relu:
	@staticmethod
	def value(x):
		return x.clip(min=0)

	@staticmethod
	def grad(x):
		return (np.sign(x) + 1) // 2


class Sigmoid:
	@staticmethod
	def value(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def grad(self, x):
		return self.value(x) * (1 - self.value(x))


class Linear:
	@staticmethod
	def value(m, x, c):
		return m * x + c

	@staticmethod
	def grad(m, x, c):
		return m


class Tanh:
	@staticmethod
	def value(a, b, x):
		return a * np.tanh(b * x)

	@staticmethod
	def grad(a, b, x):
		return a * b / np.square(np.cosh(b * x))


class Softmax:
	@staticmethod
	def value(X):
		exp_vals = np.exp(X)
		return exp_vals / (np.sum(exp_vals))

	@staticmethod
	def grad(self, X):
		return np.multiply(self.value(X), 1 - self.value(X))


if __name__ == '__main__':
	neuralNet = NeuralNet(5, [6, 2, 3, 4, 5], 'relu', 0.3)
	input = np.asarray([8, 5, 2, 3, 1, 7])
	neuralNet.forward_phase(input)
	d = np.asarray([4, 2, 3, 1, 2])
	neuralNet.backward_phase(d=d)
