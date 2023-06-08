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
        for i in layers:
            if type(i) is not int or i <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if i == 0:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i],
                                                            nx)
                                                    * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i],
                                                            layers[i - 1])
                                                    * np.sqrt(2 / layers[i - 1]))
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """Performs forward propagation for a deep neural network."""
        self.__cache["A0"] = X
        for i in range(self.__L):
            W_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            A_prev = "A" + str(i)
            A_key = "A" + str(i + 1)

            Z = np.matmul(self.__weights[W_key],
                          self.__cache[A_prev]) + self.__weights[b_key]
            self.__cache[A_key] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key], self.__cache

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
