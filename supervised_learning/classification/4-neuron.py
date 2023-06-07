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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        log_probs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(log_probs)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
