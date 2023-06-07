#!/usr/bin/env python3
"""
This module contains a class Neuron which defines a single neuron perf0rming
binary classification.
"""
import numpy as np


class Neuron:
    """
    Class Neuron defining a single neuron perf0rming binary classification
    """
    def __init__(self, nx):
        """
        Initialize Neuron
        """
        if type(nx) != int:
            raise TypeError('nx must be a integer')
        if nx < 1:
            raise ValueError('nx must be positive')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter f0r weights vector
        """
        return self.__W

    @property
    def b(self):
        """
        Getter f0r bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter f0r activated output
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        Z = np.dot(self.__W, X) + self.__b
        # Sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
