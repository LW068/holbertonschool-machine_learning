#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification"""

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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            W_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)

            if i == 0:
                self.__weights[W_key] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[W_key] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L (Number of layers)"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
