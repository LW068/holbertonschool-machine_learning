#!/usr/bin/env python3
""" Module to create a DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork class
    """
    def __init__(self, nx, layers):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not layers:
            raise TypeError('layers must be a list of positive integers')
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            W_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)

            if i == 0:
                W_value = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                W_value = np.random.randn(layers[i], layers[i - 1])
                W_value *= np.sqrt(2 / layers[i - 1])

            self.weights[W_key] = W_value
            self.weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Returns the number of layers """
        return self.__L

    @property
    def cache(self):
        """ Holds all intermediary values of the network """
        return self.__cache

    @property
    def weights(self):
        """ Holds all weights and bias """
        return self.__weights
