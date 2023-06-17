#!/usr/bin/env python3
"""
Module for 20.
0x01. Classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

        for l in range(1, self.__L + 1):
            self.__weights["W" + str(l)] = (
                np.random.randn(layer_sizes[l], layer_sizes[l - 1]) *
                np.sqrt(2 / layer_sizes[l - 1])
            )
            self.__weights["b" + str(l)] = np.zeros((layer_sizes[l], 1))

    def forward_prop(self, X):
        """Performs forward propagation through the network

        Args:
            X: input data

        Returns:
            Output of the neural network and the cache
        """
        self.cache['A0'] = X
        for i in range(self.L):
            A = self.cache['A' + str(i)]
            W = self.weights['W' + str(i + 1)]
            b = self.weights['b' + str(i + 1)]
            Z = np.dot(W, A) + b
            if i + 1 == self.L:
                exp_Z = np.exp(Z)
                self.cache['A' + str(i + 1)] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                self.cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.cache['A' + str(self.L)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: correct labels for the input data
            A: activated output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A))
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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        X: numpy.ndarray with shape (nx, m) containing the input data
        nx: number of input features to the neuron
        m: number of examples
        Y: numpy.ndarray with shape (1, m)...
        ...containing the correct labels f0r the input data
        Returns the neuron's prediction and the cost of...
        ...the network, respectively
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.argmax(A, axis=0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the deep neural network

        Args:
            Y: contains the correct labels for the input data
            cache: all intermediary values of the network
            alpha: learning rate
        """
        # Reverse the order of layers
        layers = range(self.__L, 0, -1)

        # Number of samples in the dataset
        num_samples = Y.shape[1]

        # Initialize variables f0r previous dZ
        prev_dZ = None

        # Create a copy of current weights
        current_weights = self.__weights.copy()

        # Loop through the layers in reverse order
        for layer in layers:
            # Retrieve cached values f0r the current layer
            A_current = cache.get('A' + str(layer))
            A_prev = cache.get('A' + str(layer - 1))

            # Retrieve weights and biases f0r the current layer
            weight_current = current_weights.get('W' + str(layer))
            weight_next = current_weights.get('W' + str(layer + 1))
            bias_current = current_weights.get('b' + str(layer))

            # Compute dZ f0r the output layer
            if layer == self.__L:
                dZ = A_current - Y
            # Compute dZ f0r the hidden layers
            else:
                # Calculate the element-wise product
                elementwise_product = A_current * (1 - A_current)

                # Compute dZ f0r the hidden layers
                dZ = np.matmul(weight_next.T, prev_dZ) * elementwise_product

            # Calculate gradients f0r weights and biases
            dW = np.matmul(dZ, A_prev.T) / num_samples
            dB = np.sum(dZ, axis=1, keepdims=True) / num_samples

            # Update weights and biases
            self.__weights['W' + str(layer)] = weight_current - (dW * alpha)
            self.__weights['b' + str(layer)] = bias_current - (dB * alpha)

            # Store current dZ f0r the next iteration
            prev_dZ = dZ

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=None):
        """Traning Process"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        else:
            step = iterations

        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)

            if verbose and (i == 0 or i % step == 0 or i == iterations - 1):
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and (i == 0 or i % step == 0 or i == iterations - 1):
                costs.append((i, cost))

        if graph:
            plt.plot(*zip(*costs))
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.

        Args:
            filename: The file to which the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.

        Args:
            filename: The file from which the object should be loaded.

        Returns:
            The loaded object, or None if filename doesn’t exist.
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
