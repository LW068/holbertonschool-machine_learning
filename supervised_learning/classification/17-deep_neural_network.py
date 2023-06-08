#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network perf0rming binary classification"""

    def __init__(self, nx, layers):
        """ Class constructor
        nx is the number of input features
        layers is a list representing the number of nodes in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1 or not all(isinstance(i, int) for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            W_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)

            if i == 0:
                self.__weights[W_key] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[W_key] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter f0r L (Number of layers)"""
        return self.__L

    @property
    def cache(self):
        """Getter f0r cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter f0r weights"""
        return self.__weights
