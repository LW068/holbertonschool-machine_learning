#!/usr/bin/env python3
"""
This module includes a function for updating the weights and biases
of a neural network using gradient descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i-1)]
        W = weights['W' + str(i)]
        db = np.sum(dz, axis=1, keepdims=True) / m
        dW = np.matmul(dz, A.T) / m + lambtha * W / m
        dz = np.matmul(W.T, dz) * (1 - A**2)
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db