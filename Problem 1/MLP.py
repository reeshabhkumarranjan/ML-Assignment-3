import idx2numpy as idx2numpy
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
	biases = None
	num_labels = None
	num_inputs = None

	def __init__(self, num_layers, num_nodes, activation_function, learning_rate, num_labels, num_inputs):
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
		self.biases = [None] * (num_layers)
		self.num_labels = num_labels
		self.num_inputs = num_inputs

		# initialise the weights
		for layer in range(num_layers - 1):
			self.weights[layer] = np.random.normal(loc=0, scale=1, size=(self.num_nodes[layer], self.num_nodes[layer + 1]))

		# initialise the bias for each node
		for layer in range(num_layers):
			self.biases[layer] = np.zeros((self.num_nodes[layer], 1))
		# initialize the outputs, deltas to empty
		for output in range(len(self.outputs)):
			self.outputs[output] = np.empty((num_nodes[output], 1))
			# self.deltas[output] = np.empty((num_nodes[output], 1))

	def forward_phase(self, input):
		input = input.reshape(-1, 1)
		self.outputs[0] = input
		self.outputs_derivative[0] = Relu.value(input)
		for layer in range(1, self.num_layers):
			output = np.dot(np.transpose(self.weights[layer - 1]), self.outputs[layer - 1]) + self.biases[layer]
			if layer == self.num_layers - 1:
				output = Softmax.value(output)
				self.outputs_derivative[-1] = Softmax.grad(output)
			else:
				output = Relu.value(output)
				self.outputs_derivative[layer] = Relu.grad(output)
			self.outputs[layer] = output

	def backward_phase(self, d):
		"""Call it with layer = 0"""

		d = d.reshape(-1, 1)
		# calculate deltas for the output layer
		self.deltas[-1] = np.multiply(self.outputs_derivative[-1], d - self.outputs[-1])

		# calculate deltas for previous layers and update weights
		for layer in range(self.num_layers - 2, -1, -1):
			self.weights[layer] += self.learning_rate * np.dot(self.outputs[layer], np.transpose(self.deltas[layer + 1]))
			self.deltas[layer] = np.multiply(np.dot(self.weights[layer], self.deltas[layer + 1]), self.outputs_derivative[layer])

		# update weights for all the layers
		# for layer in range(self.num_layers - 1):
		# 	pass

		# update bias for all the layers
		# for layer in range(self.num_layers):
		# 	self.biases[layer] -= self.learning_rate * np.multiply(np.sum(self.deltas[layer + 1]) * np.ones((self.num_nodes[layer], 1)), self.outputs[layer])


	def fit(self, x, y, batch_size, epochs):
		for epoch in range(epochs):
			print("Running epoch " + str(epoch) + "...")
			for row in range(x.shape[0]):
				input = x[row, :]
				d = np.zeros((num_labels, 1))
				for i in range(num_labels):
					d[i, 0] = 1 if i == y[i, 0] else 0
				self.forward_phase(input)
				self.backward_phase(d)

	def predict(self, X):
		self.forward_phase(X)
		return self.outputs[-1]

	def score(self, x_test, y_test):
		y_pred = self.predict(x_test)
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
	def grad(X):
		return np.multiply(Softmax.value(X), 1 - Softmax.value(X))


if __name__ == '__main__':
	# neuralNet = NeuralNet(5, [6, 2, 3, 4, 5], 'relu', 0.3)
	# input = np.asarray([8, 5, 2, 3, 1, 7])
	# neuralNet.forward_phase(input)
	# d = np.asarray([4, 2, 3, 1, 2])
	# neuralNet.backward_phase(d=d)
	training_image_set = idx2numpy.convert_from_file('images/train-images.idx3-ubyte')
	training_label_set = idx2numpy.convert_from_file('images/train-labels.idx1-ubyte')
	test_image_set = idx2numpy.convert_from_file('images/t10k-images.idx3-ubyte')
	test_label_set = idx2numpy.convert_from_file('images/t10k-labels.idx1-ubyte')

	training_y = np.transpose(np.asmatrix(training_label_set))
	test_y = np.transpose(np.asmatrix(test_label_set))

	training_set = np.zeros((training_image_set.shape[0], training_image_set.shape[1] ** 2))
	test_set = np.zeros((test_image_set.shape[0], test_image_set.shape[1] ** 2))

	x = training_set[:100, :]
	y = training_y[:100, :]
	num_inputs = x.shape[1]
	num_labels = 10

	neuralNet = NeuralNet(5, [num_inputs, 256, 128, 64, num_labels], 'relu', 0.3, num_labels, num_inputs)
	neuralNet.fit(x, y, 0, 50)