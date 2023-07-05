#!/usr/bin/env python3
"""
Implementation of Forward Propagation with Dropout
"""
import numpy as np


def forward_propagation_dropout(X, weights, num_layers, keep_prob):
    """
    Perform forward propagation with Dropout regularization.

    Args:
        X: Input data of shape (input_size, m).
        weights: Dictionary containing the weights and biases of the neural network.
        num_layers: Number of layers in the network.
        keep_prob: Probability that a node will be kept.

    Returns:
        cache: Dictionary containing the outputs and dropout masks of each layer.
    """
    cache = {}
    cache['A0'] = X

    for i in range(num_layers):
        Z = np.matmul(weights['W' + str(i + 1)], cache['A' + str(i)]) + weights['b' + str(i + 1)]

        if i == num_layers - 1:
            cache['A' + str(i + 1)] = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            cache['A' + str(i + 1)] = np.tanh(Z)

            dropout_mask = np.random.rand(*cache['A' + str(i + 1)].shape) < keep_prob
            cache['A' + str(i + 1)] *= dropout_mask
            cache['A' + str(i + 1)] /= keep_prob
            cache['D' + str(i + 1)] = dropout_mask

    return cache
