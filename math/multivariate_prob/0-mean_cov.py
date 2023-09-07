#!/usr/bin/env python3
"""This module contains functions to calculate the mean and covariance
of a given data set represented as a 2D numpy.ndarray.
"""

import numpy as np


def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.

    Parameters:
    - X (numpy.ndarray of shape (n, d)): The data set.

    Returns:
    - mean (numpy.ndarray of shape (1, d)): The mean of the data set.
    - cov (numpy.ndarray of shape (d, d)): The covariance matrix of the data set.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0).reshape(1, d)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / n

    return mean, cov


if __name__ == '__main__':
    import numpy as np

    np.random.seed(0)
    X = np.random.multivariate_normal(
        [12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
