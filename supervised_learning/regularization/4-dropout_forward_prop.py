#!/usr/bin/env python3
"""
Module that contains a function for forward propagation with dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
           nx: number of input features
           m: number of data points
        weights: dictionary of the weights and biases of the neural network
        L: the number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """
    cache = {}
    cache["A0"] = X
    for layer in range(1, L + 1):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]
        Z = np.matmul(W, A_prev) + b
        if layer != L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A *= D
            A /= keep_prob
            cache["D" + str(layer)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        cache["A" + str(layer)] = A
        cache["Z" + str(layer)] = Z
    return cache
