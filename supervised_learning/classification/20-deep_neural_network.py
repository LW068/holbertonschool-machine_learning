#!/usr/bin/env python3
"""
Module for 20. Evaluate DeepNeuralNetwork
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
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer_sizes = np.concatenate(([nx], layers))
        Ws = ("W" + str(i + 1) for i in range(self.__L))
        bs = ("b" + str(i + 1) for i in range(self.__L))

        for W, b, l in zip(Ws, bs, range(1, self.__L + 1)):
            self.__weights[W] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * np.sqrt(2 / layer_sizes[l - 1])
            self.__weights[b] = np.zeros((layer_sizes[l], 1))

    def forward_prop(self, X):
        """Performs forward propagation for a deep neural network."""
        self.__cache["A0"] = X
        A_prev = X
        for l in range(1, self.__L + 1):
            A = 1 / (1 + np.exp(-(np.dot(self.__weights["W" + str(l)], A_prev) + self.__weights["b" + str(l)])))
            self.__cache["A" + str(l)] = A
            A_prev = A
        return A, self.__cache

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
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
