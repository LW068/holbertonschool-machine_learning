#!/usr/bin/env python3
"""
Module that contains a function for gradient descent with dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) that contains the
           correct labels for the data
           classes: number of classes
           m: number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer
               of the neural network
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network

    Returns:
        None
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for layer in range(L, 0, -1):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.matmul(W.T, dZ)
        if layer != 1:
            dA = dA_prev * (1 - (A_prev ** 2)) * \
            (cache["D" + str(layer - 1)] / keep_prob)

        else:
            dA = dA_prev
        weights["W" + str(layer)] = weights["W" + str(layer)] - alpha * dW
        weights["b" + str(layer)] = weights["b" + str(layer)] - alpha * db
        dZ = dA
