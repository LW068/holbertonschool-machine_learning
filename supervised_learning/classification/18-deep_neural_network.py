#!/usr/bin/env python3
"""
Module for 18. DeepNeuralNetwork Forward Propagation
0x01. Classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initializes the DeepNeuralNetwork object
        Args:
            nx (int): Number of input features
            layers (list): List representing the number of nodes in each layer
        Raises:
            TypeError: If nx is not an integer or if layers is not a list
            ValueError: If nx or any element in layers is not a positive integer
        """
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

        for l in range(1, self.__L + 1):
            self.__weights["W" + str(l)] = np.random.randn(layer_sizes[l],
                                                           layer_sizes[l - 1]) * np.sqrt(2 / layer_sizes[l - 1])
            self.__weights["b" + str(l)] = np.zeros((layer_sizes[l], 1))

    @property
    def L(self):
        """
        Getter method for __L
        Returns:
            int: Number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for __cache
        Returns:
            dict: A dictionary representing the cache of the neural network
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for __weights
        Returns:
            dict: A dictionary representing the weights and biases of the neural network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                               where nx is the number of input features
                               and m is the number of examples
        Returns:
            tuple: Output of the neural network and cache, respectively
        """
        self.__cache["A0"] = X
        A = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            Z = np.matmul(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(l)] = A
        return A, self.__cache