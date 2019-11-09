import numpy as np

class Weight:
    value = 0

    def getValue(self):
        return self.value

    def setValue(self):
        return self.value

class Node:
    layer_number = 0
    input_node = False
    output_node = False
    hidden_node = False
    activation_function = "relu"
    back_connections = None
    forward_connections = None

    def __init__(self, layer_number, position, activation_function):
        self.layer_number = layer_number
        if position == 'i':
            self.input_node = True
        elif position == 'o':
            self.output_node = True
        else:
            self.hidden_node = True

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
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.weights = [None] * (num_layers - 1)
        self.outputs = [None] * (num_layers)
        self.outputs_derivative = [None] * (num_layers)
        self.deltas = [None] (num_layers)

        for layer in range(num_layers - 1):
            # self.weights[layer][0] = np.empty((num_nodes[layer + 1], 1))
            self.weights[layer] = [None] * (num_nodes[layer + 1])
            # self.deltas[layer] =
            for node in range(num_nodes[layer + 1]):
                self.weights[layer][node] = np.random.normal(loc=0, scale=1, size=(num_nodes[layer] + 1, 1))
                self.weights[layer][node][-1] = 0

        for output in range(len(self.outputs)):
            # self.outputs[output] = [None] * num_nodes[output]
            self.outputs[output] = np.empty((num_nodes[output], 1))
            self.deltas[output] = np.empty((num_nodes[output], 1))
            # self.outputs_derivative = np.empty((num_nodes[output], 1))
    def forward_phase(self, input):
        # output = input
        # self.outputs[0] = np.concatenate((input, np.ones((1,))))
        self.outputs[0] = input
        input = input.reshape(-1, 1)
        # input = np.concatenate((input, np.ones((1, 1))))
        self.outputs[0] = input.reshape((-1, 1))
        for layer in range(1, self.num_layers):
            for node in range(self.num_nodes[layer]):

                # output = np.dot(self.weights[layer - 1][node], np.transpose(self.outputs[layer - 1]))
                input = np.concatenate((self.outputs[layer - 1], np.ones((1, 1))))
                output = np.dot(np.transpose(input), self.weights[layer - 1][node])
                output = Relu().value(output)
                self.outputs[layer][node] = output
            # self.outputs[layer] = np.concatenate((self.outputs[layer], np.ones((1, 1))))
            # self.outputs[layer] = o
            self.outputs_derivative[layer] = Relu().grad(self.outputs[layer])

    def backward_phase(self, d, layer=0):

        if layer == self.num_layers - 1:
            for node in self.num_nodes[layer]:
                error_signal = d[node] - Relu().value()(self.outputs[layer][node])
                phi_dash = self.outputs_derivative[layer][node]
                delta = error_signal * phi_dash
                self.deltas[layer][node] = delta
                return

        for node in self.num_nodes[layer]:
            delta_sum = 0
            for next_node in self.num_nodes[layer + 1]:
                self.backward_phase(layer = layer + 1)
                delta_sum += self.deltas[layer + 1][next_node]
            phi_dash = self.outputs_derivative[layer][node]
            delta = delta_sum * phi_dash
            self.deltas[layer][node] = delta

            # adjust weights

    def fit(self, X, Y, batch_size, epochs):
        pass

    def predict(self, X):
        pass

    def score(self, X, Y):
        pass


class Relu:
    def value(self, x):
        return x.clip(min=0)

    def grad(self, x):
        return (np.sign(x) + 1) // 2


class Sigmoid:
    def value(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, x):
        return self.value(x) * (1 - self.value(x))


class Linear:
    def value(self, m, x, c):
        return m * x + c

    def grad(self, m, x, c):
        return m


class Tanh:
    def value(self, a, b, x):
        return a * np.tanh(b * x)

    def grad(self, a, b, x):
        return a * b / np.square(np.cosh(b * x))


class Softmax:
    def value(self, X):
        exp_vals = np.exp(X)
        return exp_vals / (np.sum(exp_vals))

    def grad(self, X):
        return np.multiply(self.value(X), 1 - self.value(X))

if __name__ == '__main__':
    neuralNet = NeuralNet(5, [6, 2, 3, 4, 5], 'relu', 0.3)
    input = np.asarray([8, 5, 2, 3, 1, 7])
    neuralNet.forward_phase(input)