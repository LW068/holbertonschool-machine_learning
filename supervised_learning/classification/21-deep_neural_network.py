#!/usr/bin/env python3
"""
Module for 20.
0x01. Classification
"""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing binary
    classification.
    """
    def __init__(self, nx, layers):
        """Initialize all the variables."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer_sizes = np.concatenate(([nx], layers))

        for l in range(1, self.__L + 1):
            self.__weights["W" + str(l)] = (
                np.random.randn(layer_sizes[l], layer_sizes[l - 1]) *
                np.sqrt(2 / layer_sizes[l - 1])
            )
            self.__weights["b" + str(l)] = np.zeros((layer_sizes[l], 1))

    def forward_prop(self, X):
        """Performs forward propagation for a deep neural network."""
        self.__cache["A0"] = X
        A_prev = X
        for l in range(1, self.__L + 1):
            A = 1 / (1 + np.exp(-(
                np.matmul(self.__weights["W" + str(l)], A_prev) +
                self.__weights["b" + str(l)]
            )))
            self.__cache["A" + str(l)] = A
            A_prev = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = (
            -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        )
        return cost

    @property
    def cache(self):
        """Getter for cache attribute."""
        return self.__cache

    @property
    def L(self):
        """Getter for L attribute."""
        return self.__L

    @property
    def weights(self):
        """Getter for weights attribute."""
        return self.__weights

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        X: numpy.ndarray with shape (nx, m) containing the input data
        nx: number of input features to the neuron
        m: number of examples
        Y: numpy.ndarray with shape (1, m)...
        ...containing the correct labels f0r the input data
        Returns the neuron's prediction and the cost of...
        ...the network, respectively
        """
        A, _ = self.forward_prop(X)  # Get the output of the network
        cost = self.cost(Y, A)  # Calculate the cost
        prediction = np.where(
            A >= 0.5, 1, 0
        )  # Apply the threshold to get the predicted labels
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""
        n_layers = range(self.__L, 0, -1)
        m = Y.shape[1]
        dZ_prev = 0
        weights = self.__weights.copy()

        for i in n_layers:
            A = cache.get('A' + str(i))
            A_prev = cache.get('A' + str(i - 1))
            weights_i = weights.get('W' + str(i))
            weights_n = weights.get('W' + str(i + 1))
            biases = weights.get('b' + str(i))
            if i == self.__L:
                dZ = A - Y
            else:
                dZ = np.matmul(weights_n.T, dZ_prev) * (A * (1 - A))
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights['W' + str(i)] = weights_i - (dW * alpha)
            self.__weights['b' + str(i)] = biases - (db * alpha)
            dZ_prev = dZ
