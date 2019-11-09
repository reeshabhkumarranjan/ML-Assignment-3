import numpy as np


class NeuralNet:
    num_layers = 2
    num_nodes = [1, 1]
    activation_function = "relu"
    learning_rate = 0.3

    def __init__(self, num_layers, num_nodes, activation_function, learning_rate):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    def fit(self, X, Y, batch_size, epochs):
        pass

    def predict(self, X):
        pass

    def score(self, X, Y):
        pass


class Relu:
    def value(self, x):
        return max(0, x)

    def grad(self, x):
        return 0 if x < 0 else 1


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
