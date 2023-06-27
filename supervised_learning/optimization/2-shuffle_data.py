#!/usr/bin/env python3
"""
This module contains the function shuffle_data...
...which shuffles the data points in two...
...matrices the same way.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X: first numpy.ndarray of shape (m, nx) to shuffle
            m: number of data points
            nx: number of features in X
        Y: second numpy.ndarray of shape (m, ny) to shuffle
            m: same number of data points as in X
            ny: number of features in Y

    Returns:
        shuffled_X, shuffled_Y: The shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y
