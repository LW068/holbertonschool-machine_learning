#!/usr/bin/env python3
"""
This module contains the function normalization_constants which calculates the normalization (standardization) constants of a matrix.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix
    Args:
        X: numpy.ndarray of shape (m, nx) to normalize
            m: number of data points
            nx: number of features
    Returns: the mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
