#!/usr/bin/env python3
"""
Module for function dropout_forward_prop
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    X is a numpy.ndarray of shape (nx, m)...
    ...containing the input data for the network
    nx is the number of input features
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function

    Returns: a dictionary containing the outputs of each layer and the dropout
             mask used on each layer
    """

    cache = {}
    cache['A0'] = X

    for i in range(L):
        # Linear Transformation
        Z = np.matmul(weights['W' + str(i + 1)],
                      cache['A' + str(i)]) + weights['b' + str(i + 1)]

        # Activation
        if i == L - 1:
            # Softmax activation for the last layer
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            # tanh activation for the other layers
            A = np.tanh(Z)
            # Dropout
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A *= D
            A /= keep_prob
            cache['D' + str(i + 1)] = D

        cache['A' + str(i + 1)] = A

    return cache
