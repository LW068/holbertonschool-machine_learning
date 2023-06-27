#!/usr/bin/env python3
"""
This module contains the function normalization_constants which calculates the normalization constants of a matrix.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
            d: number of data points
            nx: number of features
        m: numpy.ndarray of shape (nx,)...
        ...that contains the mean of all features of X
        s: numpy.ndarray of shape (nx,) that contains...
        ...the standard deviation of all features of X
    Returns: The normalized X matrix
    """
    X_norm = (X - m) / s
    return X_norm
