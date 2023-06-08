#!/usr/bin/env python3
"""
Module for 18. DeepNeuralNetwork Forward Propagation
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
        Ws = (f"W{i+1}" for i in range(self.__L))
        bs = (f"b{i+1}" for i in range(self.__L))

        for W, b, l in zip(Ws, bs, range(1, self.__L+1)):
            self.__weights[W] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2 / layer_sizes[l-1])
            self.__weights[b] = np.zeros((layer_sizes[l], 1))

    def forward_prop(self, X):
        """Performs forward propagation for a deep neural network."""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            Z = np.matmul(W, self.__cache[f"A{i-1}"]) + b
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))
        return self.__cache[f"A{self.__L}"], self.__cache

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
