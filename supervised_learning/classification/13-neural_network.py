#!/usr/bin/env python3
"""
Module containing the NeuralNetwork class for a binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork for binary classification
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for A2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (1/m) * np.matmul(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1/m) * np.matmul(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
